import os
import shutil
import logging
import time
import glob
import yaml
from tkinter import E

import blobfile as bf

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.ema import EMAHelper
from models.diffusion import Model
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import create_model, load_ckpt, load_classifier, load_model_zoo
from utils.model_schedule_util import get_model_schedule
from evaluate.fid_score import calculate_fid_given_paths

import torchvision.utils as tvu


def load_data_for_worker(base_samples, batch_size, cond_class):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if cond_class:
            label_arr = obj["arr_1"]
    buffer = []
    label_buffer = []
    while True:
        for i in range(len(image_arr)):
            buffer.append(image_arr[i])
            if cond_class:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = torch.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if cond_class:
                    res["y"] = torch.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
    
    def sample(self):
        
        map_location = self.device
        if self.args.use_model_schedule:
            model_zoo = load_model_zoo(self.config, self.device, map_location)
            model_schedule = get_model_schedule(self.config.sampling.model_schedule)
            model = None
        else:
            model = create_model(self.config)
            model = model.to(self.device)
            load_ckpt(model, self.config, map_location)
            model.eval()
            model_zoo = None
            model_schedule = None
        
        if self.config.sampling.cond_class and not self.config.model.is_upsampling:
            classifier = load_classifier(self.config, map_location)
            classifier = classifier.to(self.device)
        else:
            classifier = None

        if self.args.fid:
            self.sample_fid(model, model_zoo, model_schedule, self.args.use_model_schedule, classifier=classifier)
            print("Begin to compute FID...")
            fid = calculate_fid_given_paths((self.config.sampling.fid_stats_dir, self.args.image_folder), batch_size=self.config.sampling.fid_batch_size, device=self.device, dims=2048, num_workers=8)
            print("FID: {}".format(fid))
            np.save(os.path.join(self.args.exp, "fid"), fid)
                
            if not self.config.sampling.keep_samples:
                shutil.rmtree(self.args.image_folder)
        # elif self.args.interpolation:
        #     self.sample_interpolation(model)
        # elif self.args.sequence:
        #     self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")
        
    def dataset_generate(self):
        
        os.makedirs(self.config.dataset_generate.dataset_path, exist_ok=True)
        
        map_location = self.device
        model_zoo = load_model_zoo(self.config, self.device, map_location)
        
        if self.config.sampling.cond_class and not self.config.model.is_upsampling:
            classifier = load_classifier(self.config, map_location)
            classifier = classifier.to(self.device)
        else:
            classifier = None
            
        if self.config.dataset_generate.metric=="fid":
            for i in range(self.config.dataset_generate.data_num):
                # delete generated images
                if os.path.exists(self.args.image_folder):
                    shutil.rmtree(self.args.image_folder)
                
                # sample model schedule from search space
                if self.args.sample_type=="generalized":
                    ms_length = self.args.timesteps
                elif self.args.sample_type=="dpmsolver":
                    ms_length = self.config.dataset_generate.model_schedule.multinomial.ms_length
                else:
                    raise NotImplementedError(f"Sampler type {self.args.sample_type} is not currenty supported for (model schedule, metric) dataset generation!")
                model_schedule = get_model_schedule(self.config.dataset_generate.model_schedule, ms_length, self.args.sample_type, len(model_zoo)-1) # -1 is because null model is treated separately
                
                # calculate fid
                self.sample_fid(None, model_zoo, model_schedule, True, classifier=classifier, total_n_samples=self.config.dataset_generate.image_num)
                fid = calculate_fid_given_paths((self.config.sampling.fid_stats_dir, self.args.image_folder), batch_size=self.config.sampling.fid_batch_size, device=self.device, dims=2048, num_workers=8)
                
                # save (model schedule, fid)
                save_path = os.path.join(self.config.dataset_generate.dataset_path, f"{len(os.listdir(self.config.dataset_generate.dataset_path))}.pth")
                print(f"Model Schedule: {model_schedule} | FID: {fid}. Save to {save_path}") 
                torch.save({"ms": model_schedule, "metric": fid}, save_path)
                
                shutil.rmtree(self.args.image_folder)
                
        else:
            raise NotImplementedError(f"Currently metric {self.args.metric_type} is not supported!")
        
    def population_eval(self):
        
        map_location = self.device
        model_zoo = load_model_zoo(self.config, self.device, map_location)
        
        if self.config.sampling.cond_class and not self.config.model.is_upsampling:
            classifier = load_classifier(self.config, map_location)
            classifier = classifier.to(self.device)
        else:
            classifier = None
            
        population = torch.load(self.args.load_population)
            
        if self.config.pop_eval.metric=="fid":
            min_fid = 1e5
            min_ms = None
            for i in range(self.config.pop_eval.eval_schedule_num):
                # delete generated images
                if os.path.exists(self.args.image_folder):
                    shutil.rmtree(self.args.image_folder)
                
                model_schedule = population[i]
                
                # calculate fid
                self.sample_fid(None, model_zoo, model_schedule, True, classifier=classifier, total_n_samples=self.config.pop_eval.image_num)
                fid = calculate_fid_given_paths((self.config.sampling.fid_stats_dir, self.args.image_folder), batch_size=self.config.sampling.fid_batch_size, device=self.device, dims=2048, num_workers=8)
                logging.info(f"Model Schedule: {model_schedule}")
                logging.info(f"FID of {self.config.pop_eval.image_num} images: {fid}")
                
                # get the best ms
                if min_fid > fid:
                    min_ms = model_schedule
                    min_fid = fid
            
            logging.info(f"The best model schedule: {min_ms}")
            
            # eval the best ms
            if os.path.exists(self.args.image_folder):
                shutil.rmtree(self.args.image_folder)
            self.sample_fid(None, model_zoo, min_ms, True, classifier=classifier)
            fid = calculate_fid_given_paths((self.config.sampling.fid_stats_dir, self.args.image_folder), batch_size=self.config.sampling.fid_batch_size, device=self.device, dims=2048, num_workers=8)
            
            logging.info(f"The fid of searched schedule is {fid}")
            np.save(os.path.join(self.args.exp, "fid"), fid)
            
            if not self.config.sampling.keep_samples:
                shutil.rmtree(self.args.image_folder)
        else:
            raise NotImplementedError(f"Currently metric {self.args.metric_type} is not supported!")
        

    def sample_fid(self, model, model_zoo:list=None, model_schedule:list=None, use_model_schedule=False, classifier=None, total_n_samples=None):
        config = self.config
        if total_n_samples is None:
            total_n_samples = config.sampling.fid_total_samples
        if total_n_samples % config.sampling.batch_size != 0:
            raise ValueError("Total samples for sampling must be divided exactly by config.sampling.batch_size, but got {} and {}".format(total_n_samples, config.sampling.batch_size))
        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)
        if len(glob.glob(f"{self.args.image_folder}/*.png")) == total_n_samples:
            return
        else:
            n_rounds = total_n_samples // config.sampling.batch_size
        img_id = 0

        if self.config.model.is_upsampling:
            base_samples_total = load_data_for_worker(self.args.base_samples, config.sampling.batch_size, config.sampling.cond_class)
        
        # use fixed noise for evaluation to reduce randomness
        if self.config.fixed_noise.enable:
            if  getattr(self, "fixed_noise", None) is None:
                self.fixed_noise = torch.load(self.config.fixed_noise.path).to(self.device)
            noise_idx = 0
            fixed_noise = self.fixed_noise
            assert total_n_samples<=len(fixed_noise), "The length of noise must larger than the number of samples"

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                # torch.cuda.synchronize()
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()

                n = config.sampling.batch_size
                
                if self.config.fixed_noise.enable:
                    x = fixed_noise[noise_idx:noise_idx+n]
                    noise_idx += n
                else:
                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )

                if self.config.model.is_upsampling:
                    base_samples = next(base_samples_total)
                else:
                    base_samples = None

                x, classes = self.sample_image(x, model, model_zoo=model_zoo, model_schedule=model_schedule, use_model_schedule=use_model_schedule, classifier=classifier, base_samples=base_samples)

                # end.record()
                # torch.cuda.synchronize()
                # t_list.append(start.elapsed_time(end))
                x = inverse_data_transform(config, x)
                for i in range(x.shape[0]):
                    if classes is None:
                        path = os.path.join(self.args.image_folder, f"{img_id}.png")
                    else:
                        path = os.path.join(self.args.image_folder, f"{img_id}_{int(classes.cpu()[i])}.png")
                    tvu.save_image(x.cpu()[i], path)
                    img_id += 1
        # # Remove the time evaluation of the first batch, because it contains extra initializations
        # print('time / batch', np.mean(t_list[1:]) / 1000., 'std', np.std(t_list[1:]) / 1000.)

    def sample_sequence(self, model, model_zoo:list=None, model_schedule:list=None, use_model_schedule=False, classifier=None):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, model_zoo=model_zoo, model_schedule=model_schedule, use_model_schedule=use_model_schedule, last=False, classifier=classifier)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model, model_zoo:list=None, model_schedule:list=None, use_model_schedule=False):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model, model_zoo=model_zoo, model_schedule=model_schedule, use_model_schedule=use_model_schedule))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, model_zoo:list=None, model_schedule:list=None, use_model_schedule=False, last=True, classifier=None, base_samples=None):
        assert last
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        classifier_scale = self.config.sampling.classifier_scale if self.args.scale is None else self.args.scale
        if self.config.sampling.cond_class:
            if self.args.fixed_class is None:
                classes = torch.randint(low=0, high=self.config.data.num_classes, size=(x.shape[0],)).to(x.device)
            else:
                classes = torch.randint(low=self.args.fixed_class, high=self.args.fixed_class + 1, size=(x.shape[0],)).to(x.device)
        else:
            classes = None
        
        if base_samples is None:
            if classes is None:
                model_kwargs = {}
            else:
                model_kwargs = {"y": classes}
        else:
            model_kwargs = {"y": base_samples["y"], "low_res": base_samples["low_res"]}

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps if model_schedule is None else self.num_timesteps // len(model_schedule)
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            
            if model_schedule is not None:
                active_step_idx = np.where(np.array(model_schedule)!=0)[0]
                model_schedule = np.array(model_schedule)[active_step_idx].tolist()
                seq = np.array(seq)[active_step_idx].tolist()
                if len(model_schedule)!=len(seq):
                    raise ValueError("The length of model schedule must be equal to the number of timesteps!")
            
            from functions.denoising import generalized_steps
            def model_fn(x, t, model_idx, **model_kwargs):
                if use_model_schedule:
                    out = model_zoo[model_schedule[model_idx]](x, t, **model_kwargs) # Use the models in the model schedule to do inference
                    print(f"Timestep {t[0].item()} | Use model loaded from {model_zoo[model_schedule[model_idx]].ckpt_path}") 
                else:
                    out = model(x, t, **model_kwargs) # Use a single model to do inferencne
                    print(f"Timestep {t[0].item()} | Use model loaded from {model.ckpt_path}")
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = generalized_steps(x, seq, model_fn, self.betas, eta=self.args.eta, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            skip_step_idx = np.where(np.array(model_schedule)==0)[0]
            model_schedule = np.array(model_schedule)[skip_step_idx].tolist()
            seq = np.array(seq)[skip_step_idx].tolist()
            from functions.denoising import ddpm_steps
            def model_fn(x, t, model_idx, **model_kwargs):
                if use_model_schedule:
                    if len(model_zoo)!=len(seq):
                        raise ValueError("The length of model schedule must be equal to the number of timesteps!")
                    out = model_zoo[model_schedule[model_idx]](x, t, **model_kwargs) # Use the models in the model schedule to do inference
                    print(f"Timestep {t} | Use model loaded from {model_zoo[model_schedule[model_idx]].ckpt_path}")
                else:
                    out = model(x, t, **model_kwargs) # Use a single model to do inferencne
                    print(f"Timestep {t} | Use model loaded from {model.ckpt_path}")
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = ddpm_steps(x, seq, model_fn, self.betas, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]
        elif self.args.sample_type in ["dpmsolver", "dpmsolver++"]:
            if model_schedule is not None and self.args.sample_type=="dpmsolver++":
                raise NotImplementedError("Currently we don't support sampling with model schedule and dpmsolver++ simultaneously. We will make it done soon.")
            from dpm_solver.sampler import NoiseScheduleVP, model_wrapper, DPM_Solver
            def model_fn(x, t, model_idx, **model_kwargs):
                if use_model_schedule:
                    out = model_zoo[model_schedule[model_idx]](x, t, **model_kwargs) # Use the models in the model schedule to do inference
                    print(f"Use model from {model_zoo[model_schedule[model_idx]].ckpt_path}") 
                else:
                    out = model(x, t, **model_kwargs) # Use a single model to do inferencne
                    print(f"Use model from {model.ckpt_path}") 
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]

            noise_schedule = NoiseScheduleVP(schedule='linear', betas=self.betas)
            model_fn_continuous = model_wrapper( 
                model_fn,
                noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            dpm_solver = DPM_Solver(
                model_fn_continuous,
                noise_schedule,
                algorithm_type=self.args.sample_type,
                correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
            )
            step = self.args.timesteps if not self.args.use_model_schedule else (torch.tensor(model_schedule)!=0).sum().item()
            x = dpm_solver.sample(
                x,
                steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                model_schedule=model_schedule,
                order=self.args.dpm_solver_order,
                skip_type=self.args.skip_type,
                method=self.args.dpm_solver_method,
                lower_order_final=self.args.lower_order_final,
                denoise_to_zero=self.args.denoise,
                solver_type=self.args.dpm_solver_type,
                atol=self.args.dpm_solver_atol,
                rtol=self.args.dpm_solver_rtol,
                t_end=1e-4 if step>=15 else 1e-3 
            )
            # x = x.cpu()
        else:
            raise NotImplementedError
        return x, classes

    def test(self):
        pass
