import os, hashlib
import requests
import torch
import yaml
from utils.config_util import dict2namespace
from tqdm import tqdm

from models.ema import EMAHelper
from models.diffusion import Model
from models.simple_diffusion import SimpleNet 
from models.improved_ddpm.unet import UNetModel as ImprovedDDPM_Model
from models.guided_diffusion.unet import UNetModel as GuidedDiffusion_Model
from models.guided_diffusion.unet import EncoderUNetModel as GuidedDiffusion_Classifier
from models.guided_diffusion.unet import SuperResModel as GuidedDiffusion_SRModel

URL_MAP = {
    "cifar10": "https://heibox.uni-heidelberg.de/f/869980b53bf5416c8a28/?dl=1",
    "ema_cifar10": "https://heibox.uni-heidelberg.de/f/2e4f01e2d9ee49bab1d5/?dl=1",
    "lsun_bedroom": "https://heibox.uni-heidelberg.de/f/f179d4f21ebc4d43bbfe/?dl=1",
    "ema_lsun_bedroom": "https://heibox.uni-heidelberg.de/f/b95206528f384185889b/?dl=1",
    "lsun_cat": "https://heibox.uni-heidelberg.de/f/fac870bd988348eab88e/?dl=1",
    "ema_lsun_cat": "https://heibox.uni-heidelberg.de/f/0701aac3aa69457bbe34/?dl=1",
    "lsun_church": "https://heibox.uni-heidelberg.de/f/2711a6f712e34b06b9d8/?dl=1",
    "ema_lsun_church": "https://heibox.uni-heidelberg.de/f/44ccb50ef3c6436db52e/?dl=1",
}
CKPT_MAP = {
    "cifar10": "diffusion_cifar10_model/model-790000.ckpt",
    "ema_cifar10": "ema_diffusion_cifar10_model/model-790000.ckpt",
    "lsun_bedroom": "diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "ema_lsun_bedroom": "ema_diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "lsun_cat": "diffusion_lsun_cat_model/model-1761000.ckpt",
    "ema_lsun_cat": "ema_diffusion_lsun_cat_model/model-1761000.ckpt",
    "lsun_church": "diffusion_lsun_church_model/model-4432000.ckpt",
    "ema_lsun_church": "ema_diffusion_lsun_church_model/model-4432000.ckpt",
}
MD5_MAP = {
    "cifar10": "82ed3067fd1002f5cf4c339fb80c4669",
    "ema_cifar10": "1fa350b952534ae442b1d5235cce5cd3",
    "lsun_bedroom": "f70280ac0e08b8e696f42cb8e948ff1c",
    "ema_lsun_bedroom": "1921fa46b66a3665e450e42f36c2720f",
    "lsun_cat": "bbee0e7c3d7abfb6e2539eaf2fb9987b",
    "ema_lsun_cat": "646f23f4821f2459b8bafc57fd824558",
    "lsun_church": "eb619b8a5ab95ef80f94ce8a5488dae3",
    "ema_lsun_church": "fdc68a23938c2397caba4a260bc2445f",
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root=None, check=False):
    if 'church_outdoor' in name:
        name = name.replace('church_outdoor', 'church')
    # Modify the path when necessary
    cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/ddpm_ckpt"))
    assert name in URL_MAP
    root = (
        root
        if root is not None
        else os.path.join(cachedir, "diffusion_models_converted")
    )
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

def create_model(config):
    if config.model.model_type == 'improved_ddpm':
        model = ImprovedDDPM_Model(
            in_channels=config.model.in_channels,
            model_channels=config.model.model_channels,
            out_channels=config.model.out_channels,
            num_res_blocks=config.model.num_res_blocks,
            attention_resolutions=config.model.attention_resolutions,
            dropout=config.model.dropout,
            channel_mult=config.model.channel_mult,
            conv_resample=config.model.conv_resample,
            dims=config.model.dims,
            use_checkpoint=config.model.use_checkpoint,
            num_heads=config.model.num_heads,
            num_heads_upsample=config.model.num_heads_upsample,
            use_scale_shift_norm=config.model.use_scale_shift_norm
        )
    elif config.model.model_type == "guided_diffusion":
        if config.model.is_upsampling:
            model = GuidedDiffusion_SRModel(
                image_size=config.model.large_size,
                in_channels=config.model.in_channels,
                model_channels=config.model.model_channels,
                out_channels=config.model.out_channels,
                num_res_blocks=config.model.num_res_blocks,
                attention_resolutions=config.model.attention_resolutions,
                dropout=config.model.dropout,
                channel_mult=config.model.channel_mult,
                conv_resample=config.model.conv_resample,
                dims=config.model.dims,
                num_classes=config.model.num_classes,
                use_checkpoint=config.model.use_checkpoint,
                use_fp16=config.model.use_fp16,
                num_heads=config.model.num_heads,
                num_head_channels=config.model.num_head_channels,
                num_heads_upsample=config.model.num_heads_upsample,
                use_scale_shift_norm=config.model.use_scale_shift_norm,
                resblock_updown=config.model.resblock_updown,
                use_new_attention_order=config.model.use_new_attention_order,
            )
        else:
            model = GuidedDiffusion_Model(
                image_size=config.model.image_size,
                in_channels=config.model.in_channels,
                model_channels=config.model.model_channels,
                out_channels=config.model.out_channels,
                num_res_blocks=config.model.num_res_blocks,
                attention_resolutions=config.model.attention_resolutions,
                dropout=config.model.dropout,
                channel_mult=config.model.channel_mult,
                conv_resample=config.model.conv_resample,
                dims=config.model.dims,
                num_classes=config.model.num_classes,
                use_checkpoint=config.model.use_checkpoint,
                use_fp16=config.model.use_fp16,
                num_heads=config.model.num_heads,
                num_head_channels=config.model.num_head_channels,
                num_heads_upsample=config.model.num_heads_upsample,
                use_scale_shift_norm=config.model.use_scale_shift_norm,
                resblock_updown=config.model.resblock_updown,
                use_new_attention_order=config.model.use_new_attention_order,
            )
    elif config.model.model_type == "SimpleDiffusion":
        model = SimpleNet(config) 
    else:
        model = Model(config)
        
    return model

def load_ckpt(model, config, map_location):
    if "ckpt_dir" in config.model.__dict__.keys():
        ckpt_dir = os.path.expanduser(config.model.ckpt_dir)
        states = torch.load(
            ckpt_dir,
            map_location=map_location
        )
        # states = {f"module.{k}":v for k, v in states.items()}
        if config.model.model_type == 'improved_ddpm' or config.model.model_type == 'guided_diffusion':
            model.load_state_dict(states, strict=True)
            if config.model.use_fp16:
                model.convert_to_fp16()
        else:
            # TODO: FIXME
            # model = torch.nn.DataParallel(model)
            # model.load_state_dict(states[0], strict=True)
            model.load_state_dict(states[0], strict=True)
        model.ckpt_path = ckpt_dir

        if config.model.ema: # for celeba 64x64 in DDIM
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
            print(f"Load checkpoint from {ckpt_dir} and use EMA")
        else:
            ema_helper = None
            print(f"Load checkpoint from {ckpt_dir}")
    else:
        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif config.data.dataset == "LSUN":
            name = f"lsun_{config.data.category}"
        else:
            raise ValueError
        ckpt = get_ckpt_path(f"ema_{name}")
        print("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=map_location))

def load_classifier(config, map_location):
    classifier = GuidedDiffusion_Classifier(
        image_size=config.classifier.image_size,
        in_channels=config.classifier.in_channels,
        model_channels=config.classifier.model_channels,
        out_channels=config.classifier.out_channels,
        num_res_blocks=config.classifier.num_res_blocks,
        attention_resolutions=config.classifier.attention_resolutions,
        channel_mult=config.classifier.channel_mult,
        use_fp16=config.classifier.use_fp16,
        num_head_channels=config.classifier.num_head_channels,
        use_scale_shift_norm=config.classifier.use_scale_shift_norm,
        resblock_updown=config.classifier.resblock_updown,
        pool=config.classifier.pool
    )
    ckpt_dir = os.path.expanduser(config.classifier.ckpt_dir)
    states = torch.load(
        ckpt_dir,
        map_location=map_location,
    )
    # states = {f"module.{k}":v for k, v in states.items()}
    classifier.load_state_dict(states, strict=True)
    print(f"Load classifier checkpoint from {ckpt_dir}")
    if config.classifier.use_fp16:
        classifier.convert_to_fp16()
        # classifier.module.convert_to_fp16()
    classifier.eval()
    
    return classifier
    
def load_model_zoo(config, device, map_location):
    model_zoo = [None]
    if config.sampling.model_zoo.load_type=="zoo":
        configs = [os.path.join(config.sampling.model_zoo.path, p, "config.yml") for p in os.listdir(config.sampling.model_zoo.path) if os.path.exists(os.path.join(config.sampling.model_zoo.path, p, "config.yml"))]
    elif config.sampling.model_zoo.load_type=="single":
        configs = config.sampling.model_zoo.configs
    else:
        raise NotImplementedError("Currently only support: 1. (config.model_schedule.load_type = zoo) load all models in a dir as the model zoo; \
            2. (config.model_schedule.load_type = single) load every model in the model zoo independently.")
    for config_path in configs:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config = dict2namespace(config)
        model = create_model(config)
        model.eval()
        model = model.to(device)
        load_ckpt(model, config, map_location)
        model_zoo.append(model)
    
    return model_zoo
