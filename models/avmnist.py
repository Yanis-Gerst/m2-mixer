from abc import ABC, abstractmethod
from copy import deepcopy
from os import path

import numpy as np
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from modules.gradblend import GradBlend
from modules.losses import EDLMSELoss
from modules.train_test_module import AbstractTrainTestModule
from modules.mixer import MLPMixer
import torch
from typing import List, Any, Optional
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torch.nn.functional import softplus
import modules
from octopy.src.uncertainty_quantification.evidential.output_extractor.exponential_evidences import ExponentialEvidences
from octopy.src.uncertainty_quantification.evidential.quantification.dirichlet import Dirichlet
from octopy.src.uncertainty_quantification.loss.loss_function import EDLLossMeanWrapper
from octopy.src.logger.logger import MetricsLogger
import os

try:
    from softadapt import LossWeightedSoftAdapt
except ModuleNotFoundError:
    print('Warning: Could not import softadapt. LossWeightedSoftAdapt will not be available.')
    LossWeightedSoftAdapt = None


def activation_function(h):
    # # Compute log(1e13) accurately
    # log1e13 = 13 * \
    #     torch.log(torch.tensor(10.0, dtype=h.dtype, device=h.device))

    # # Numerator in log-space
    # numerator = h + log1e13

    # # Denominator in log-space using logaddexp for numerical stability
    # denominator = torch.logaddexp(h, log1e13)

    # # Compute the log of the function
    # log_f = numerator - denominator

    # Exponentiate to get the final result
    return torch.exp(torch.clamp(h, 0, 10))


class AbstractAVMnistMixer(AbstractTrainTestModule, ABC):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AbstractAVMnistMixer, self).__init__(optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = None
        self.classifier = None

    def shared_step(self, batch, **kwargs):
        logits = self.get_logits(batch)
        labels = batch['label']
        loss = self.criterion(logits, labels)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)

        return {
            'preds': preds,
            'labels': labels,
            'loss': loss
        }

    @abstractmethod
    def get_logits(self, batch):
        raise NotImplementedError

    def setup_criterion(self) -> torch.nn.Module:
        return CrossEntropyLoss()

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                            f1m=F1Score(task="multiclass",
                                        num_classes=10, average='macro'),
                            prec_m=Precision(
                                task="multiclass", num_classes=10, average='macro'),
                            rec_m=Recall(task="multiclass",
                                         num_classes=10, average='macro'),
                            f1mi=F1Score(task="multiclass",
                                         num_classes=10, average='micro'),
                            prec_mi=Precision(
                                task="multiclass", num_classes=10, average='micro'),
                            rec_mi=Recall(task="multiclass", num_classes=10, average='micro'))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                          f1m=F1Score(task="multiclass",
                                      num_classes=10, average='macro'),
                          prec_m=Precision(task="multiclass",
                                           num_classes=10, average='macro'),
                          rec_m=Recall(task="multiclass",
                                       num_classes=10, average='macro'),
                          f1mi=F1Score(task="multiclass",
                                       num_classes=10, average='micro'),
                          prec_mi=Precision(
                              task="multiclass", num_classes=10, average='micro'),
                          rec_mi=Recall(task="multiclass", num_classes=10, average='micro'))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                           f1m=F1Score(task="multiclass",
                                       num_classes=10, average='macro'),
                           prec_m=Precision(task="multiclass",
                                            num_classes=10, average='macro'),
                           rec_m=Recall(task="multiclass",
                                        num_classes=10, average='macro'),
                           f1mi=F1Score(task="multiclass",
                                        num_classes=10, average='micro'),
                           prec_mi=Precision(
                               task="multiclass", num_classes=10, average='micro'),
                           rec_mi=Recall(task="multiclass", num_classes=10, average='micro'))

        return [train_scores, val_scores, test_scores]

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class AVMnistImageMixer(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistImageMixer, self).__init__(
            model_cfg, optimizer_cfg, **kwargs)
        self.model = MLPMixer(**model_cfg.modalities.image,
                              dropout=model_cfg.dropout)
        self.classifier = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                          model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        image = batch['image']
        labels = batch['label']
        logits = self.model(image)
        logits = self.classifier(logits.mean(dim=1))
        return logits


class AVMnistAudioMixer(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistAudioMixer, self).__init__(
            model_cfg, optimizer_cfg, **kwargs)
        self.model = MLPMixer(**model_cfg.modalities.audio,
                              dropout=model_cfg.dropout)
        self.classifier = torch.nn.Linear(model_cfg.modalities.audio.hidden_dim,
                                          model_cfg.modalities.classification.num_classes)

    def get_logits(self, batch):
        audio = batch['audio']
        labels = batch['label']
        logits = self.model(audio)
        logits = self.classifier(logits.mean(dim=1))
        return logits


class AVMnistMixer(AbstractAVMnistMixer):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, **kwargs):
        super(AVMnistMixer, self).__init__(model_cfg, optimizer_cfg, **kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.mute = model_cfg.get('mute', None)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = modules.get_block_by_name(
            **image_config, dropout=dropout)
        self.audio_mixer = modules.get_block_by_name(
            **audio_config, dropout=dropout)
        self.fusion_function = modules.get_fusion_by_name(
            **model_cfg.modalities.multimodal)
        num_patches = self.fusion_function.get_output_shape(self.image_mixer.num_patch, self.audio_mixer.num_patch,
                                                            dim=1)
        self.fusion_mixer = modules.get_block_by_name(
            **multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier = modules.get_classifier_by_name(
            **model_cfg.modalities.classification)

    def get_logits(self, batch):
        image = batch['image']
        audio = batch['audio']

        if self.mute == 'image':
            image = torch.zeros_like(image)
        elif self.mute == 'audio':
            audio = torch.zeros_like(audio)

        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        # fuse modalities
        fused_moalities = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(fused_moalities)

        # logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
        audio_logits = audio_logits.reshape(
            audio_logits.shape[0], -1, audio_logits.shape[-1])
        image_logits = image_logits.reshape(
            image_logits.shape[0], -1, image_logits.shape[-1])

        # get logits for each modality
        logits = self.classifier(logits)

        return logits


class AVMnistMixerMultiLoss(AbstractTrainTestModule):
    def __init__(self, model_cfg: DictConfig, optimizer_cfg: DictConfig, annealing_step, **kwargs):
        self.annealing_step = annealing_step
        super(AVMnistMixerMultiLoss, self).__init__(
            optimizer_cfg, log_confusion_matrix=True, **kwargs)
        self.modalities_freezed = False
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_patience = optimizer_cfg.pop('scheduler_patience', 5)
        self.checkpoint_path = None
        self.mute = model_cfg.get('mute', None)
        self.freeze_modalities_on_epoch = model_cfg.get(
            'freeze_modalities_on_epoch', None)
        self.random_modality_muting_on_freeze = model_cfg.get(
            'random_modality_muting_on_freeze', False)
        self.muting_probs = model_cfg.get('muting_probs', None)
        image_config = model_cfg.modalities.image
        audio_config = model_cfg.modalities.audio
        multimodal_config = model_cfg.modalities.multimodal
        dropout = model_cfg.get('dropout', 0.0)
        self.image_mixer = modules.get_block_by_name(
            **image_config, dropout=dropout)
        self.audio_mixer = modules.get_block_by_name(
            **audio_config, dropout=dropout)
        self.fusion_function = modules.get_fusion_by_name(
            **model_cfg.modalities.multimodal)
        num_patches = self.fusion_function.get_output_shape(self.image_mixer.num_patch, self.audio_mixer.num_patch,
                                                            dim=1)
        self.fusion_mixer = modules.get_block_by_name(
            **multimodal_config, num_patches=num_patches, dropout=dropout)
        self.classifier_image = torch.nn.Linear(model_cfg.modalities.image.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_audio = torch.nn.Linear(model_cfg.modalities.audio.hidden_dim,
                                                model_cfg.modalities.classification.num_classes)
        self.classifier_fusion = modules.get_classifier_by_name(
            **model_cfg.modalities.classification)

        self.train_evidences = {"image": [], "audio": [], "all": []}
        self.test_evidences = {"image": [], "audio": [], "all": []}
        # self.image_criterion = CrossEntropyLoss()
        # self.audio_criterion = CrossEntropyLoss()
        # self.fusion_criterion = CrossEntropyLoss()
        self.num_classes = 10
        self.evidences_collector = ExponentialEvidences()
        self.uncertainty_quantification = Dirichlet(self.num_classes)
        self.image_criterion = EDLLossMeanWrapper(self.num_classes, self.annealing_step, torch.digamma, self.evidences_collector)
        self.audio_criterion = EDLLossMeanWrapper(self.num_classes, self.annealing_step, torch.digamma, self.evidences_collector)
        self.fusion_criterion = EDLLossMeanWrapper(self.num_classes, self.annealing_step, torch.digamma, self.evidences_collector)
        # self.image_criterion = EDLMSELoss(self.num_classes, self.annealing_step, self.device)
        # self.audio_criterion = EDLMSELoss(self.num_classes, self.annealing_step, self.device)
        # self.fusion_criterion = EDLMSELoss(self.num_classes, self.annealing_step, self.device)
        self.fusion_loss_weight = model_cfg.get('fusion_loss_weight', 1.0 / 3)
        self.fusion_loss_change = model_cfg.get('fusion_loss_change', 0)
        self.loss_change_epoch = model_cfg.get('loss_change_epoch', 0)
        self.use_softadapt = model_cfg.get('use_softadapt', False)
        self.metrics_logger = MetricsLogger()
        self.image_criterion_history = []
        self.audio_criterion_history = []
        self.fusion_criterion_history = []
        if self.use_softadapt:
            if LossWeightedSoftAdapt is None:
                print('SoftAdapt is not installed! Hence, will not be used!')
                self.use_softadapt = False
            else:
                self.image_criterion_history = []
                self.audio_criterion_history = []
                self.fusion_criterion_history = []
                self.loss_weights = torch.tensor(
                    [1.0 / 3, 1.0 / 3, 1.0 / 3], device=self.device)
                self.update_loss_weights_per_epoch = model_cfg.get(
                    'update_loss_weights_per_epoch', 6)
                self.softadapt = LossWeightedSoftAdapt(
                    beta=-0.1, accuracy_order=self.update_loss_weights_per_epoch - 1)
        self.use_gradblend = model_cfg.get('gradblend', False)
        if self.use_gradblend:
            self.gb_update_freq = model_cfg.get('gb_update_freq', 20)
            self.gb_weights = None
            self.gradblend = None
            self.gb_train_loader = None
            self.gb_val_loader = None

    def on_train_epoch_start(self) -> None:
        if self.use_gradblend and self.current_epoch % self.gb_update_freq == 0:
            encoders = [deepcopy(self.audio_mixer), deepcopy(self.image_mixer)]
            heads = [deepcopy(self.classifier_audio),
                     deepcopy(self.classifier_image)]
            if (self.gb_val_loader is None) or (self.gb_train_loader is None):
                ds = self.trainer.train_dataloader.dataset.datasets
                ds_train = Subset(ds, range(int(len(ds) * 0.1), len(ds)))
                ds_val = Subset(ds, range(int(len(ds) * 0.1)))
                bs = self.trainer.train_dataloader.loaders.batch_size
                self.gb_train_loader = DataLoader(
                    ds_train, batch_size=bs, shuffle=True)
                self.gb_val_loader = DataLoader(
                    ds_val, batch_size=bs, shuffle=True)
            self.gradblend = GradBlend(self, encoders, heads, deepcopy(self.fusion_mixer),
                                       deepcopy(self.classifier_fusion),
                                       nn.CrossEntropyLoss, self.gb_train_loader, self.gb_val_loader)
            self.gb_weights = self.gradblend.get_weights()
            print("GradBlend weights:", self.gb_weights)

    def shared_step(self, batch, **kwargs):
        # Load data

        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        if kwargs.get('mode', None) == 'train':
            if self.freeze_modalities_on_epoch is not None and (self.current_epoch == self.freeze_modalities_on_epoch) \
                    and not self.modalities_freezed:
                self._freeze_modalities()
            if self.random_modality_muting_on_freeze and (self.current_epoch >= self.freeze_modalities_on_epoch):
                self.mute = np.random.choice(['image', 'audio', 'multimodal'], p=[self.muting_probs['image'],
                                                                                  self.muting_probs['audio'],
                                                                                  self.muting_probs['multimodal']])

            if self.mute != 'multimodal':
                if self.mute == 'image':
                    image = torch.zeros_like(image)
                elif self.mute == 'audio':
                    audio = torch.zeros_like(audio)

        # get modality encodings from feature extractors
        # Embeddings here
        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        # fuse modalities
        fused_moalities = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(fused_moalities)

        # logits = logits.reshape(logits.shape[0], -1, logits.shape[-1])
        audio_logits = audio_logits.reshape(
            audio_logits.shape[0], -1, audio_logits.shape[-1])
        image_logits = image_logits.reshape(
            image_logits.shape[0], -1, image_logits.shape[-1])

        # get logits for each modality
        image_logits = self.classifier_image(image_logits.mean(dim=1))
        audio_logits = self.classifier_audio(audio_logits.mean(dim=1))
        logits = self.classifier_fusion(logits)

        # compute losses
        # loss_image = self.image_criterion(image_logits, labels)
        # loss_audio = self.audio_criterion(audio_logits, labels)
        # loss_fusion = self.fusion_criterion(logits, labels)
        loss_image = self.image_criterion(
            image_logits, labels, self.current_epoch)
        loss_audio = self.audio_criterion(
            audio_logits, labels, self.current_epoch)
        loss_fusion = self.fusion_criterion(logits, labels, self.current_epoch)

        print(loss_fusion, "YO loss fusion")

        if self.use_softadapt:
            loss = self.loss_weights[0] * loss_image + self.loss_weights[1] * loss_audio + self.loss_weights[
                2] * loss_fusion
        elif (
            self.use_gradblend
            and self.gb_weights is not None
        ):
            loss = self.gb_weights[2] * loss_fusion + self.gb_weights[1] * loss_image + self.gb_weights[
                0] * loss_audio
        else:
            ow = (1 - self.fusion_loss_weight) / 2
            loss = (self.fusion_loss_weight * loss_fusion +
                    ow * loss_image + ow * loss_audio) * 3
        if self.modalities_freezed and kwargs.get('mode', None) == 'train':
            loss = loss_fusion

        # get predictions
        image_evidences = self.evidences_collector(image_logits)
        audio_evidences = self.evidences_collector(audio_logits)
        evidences = self.evidences_collector(logits)

        # preds = torch.softmax(logits, dim=1).argmax(dim=1)
        # preds_image = torch.softmax(image_logits, dim=1).argmax(dim=1)
        # preds_audio = torch.softmax(audio_logits, dim=1).argmax(dim=1)

        preds = evidences.argmax(dim=1)
        preds_image = image_evidences.argmax(dim=1)
        preds_audio = audio_evidences.argmax(dim=1)

        image_epidemic_uncertainty, image_aleatoric_uncertainty = self.uncertainty_quantification(
            image_evidences)
        audio_epidemic_uncertainty, audio_aleatoric_uncertainty = self.uncertainty_quantification(
            audio_evidences)
        epidemic_uncertainty, aleatoric_uncertainty = self.uncertainty_quantification(
            evidences)

        print(
            f" WDSYO Uncertainty epidemic: {epidemic_uncertainty}, aleatoric: {aleatoric_uncertainty}, preds: {evidences.mean()}")
        return {
            'preds': preds,
            'preds_image': preds_image,
            'preds_audio': preds_audio,
            "evidences": evidences,
            "image_evidences": image_evidences,
            "audio_evidences": audio_evidences,
            'labels': labels,
            'loss': loss,
            'loss_image': loss_image,
            'loss_audio': loss_audio,
            'loss_fusion': loss_fusion,
            'image_logits': image_logits,
            'audio_logits': audio_logits,
            'logits': logits,
            "aleatoric_uncertainty": torch.tensor(aleatoric_uncertainty),
            "epidemic_uncertainty": torch.tensor(epidemic_uncertainty),
            "image_epidemic_uncertainty": torch.tensor(image_epidemic_uncertainty),
            "image_aleatoric_uncertainty": torch.tensor(image_aleatoric_uncertainty),
            "audio_epidemic_uncertainty": torch.tensor(audio_epidemic_uncertainty),
            "audio_aleatoric_uncertainty": torch.tensor(audio_aleatoric_uncertainty),
        }

    def _freeze_modalities(self):
        print('Freezing modalities')
        for param in self.image_mixer.parameters():
            param.requires_grad = False
        for param in self.audio_mixer.parameters():
            param.requires_grad = False
        for param in self.classifier_image.parameters():
            param.requires_grad = False
        for param in self.classifier_audio.parameters():
            param.requires_grad = False
        self.modalities_freezed = True

    def on_train_epoch_end(self) -> None:
        outputs = self.train_step_outputs
        super().on_train_epoch_end()
        wandb.log({'train_loss_image': torch.stack(
            [x['loss_image'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_audio': torch.stack(
            [x['loss_audio'] for x in outputs]).mean().item()})
        wandb.log({'train_loss_fusion': torch.stack(
            [x['loss_fusion'] for x in outputs]).mean().item()})
        self.log('train_loss_fusion', torch.stack(
            [x['loss_fusion'] for x in outputs]).mean().item())

        self.train_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        super().on_validation_epoch_end()
        val_loss_fusion = torch.stack(
            [x['loss_fusion'] for x in outputs]).mean().item()
        self.log('val_loss_fusion', val_loss_fusion)
        wandb.log({'val_loss_fusion': val_loss_fusion})
        if self.current_epoch >= self.loss_change_epoch:
            self.fusion_loss_weight = min(
                1, self.fusion_loss_weight + self.fusion_loss_change)
        if self.use_softadapt:
            self.image_criterion_history.append(torch.stack(
                [x['loss_image'] for x in outputs]).mean().item())
            self.audio_criterion_history.append(torch.stack(
                [x['loss_audio'] for x in outputs]).mean().item())
            self.fusion_criterion_history.append(torch.stack(
                [x['loss_fusion'] for x in outputs]).mean().item())
            wandb.log({'loss_weight_image': self.loss_weights[0].item()})
            wandb.log({'loss_weight_audio': self.loss_weights[1].item()})
            wandb.log({'loss_weight_fusion': self.loss_weights[2].item()})
            wandb.log({'val_loss_image': self.image_criterion_history[-1]})
            wandb.log({'val_loss_audio': self.audio_criterion_history[-1]})
            wandb.log({'val_loss_fusion': self.fusion_criterion_history[-1]})
            self.log('val_loss_fusion', self.fusion_criterion_history[-1])

            if self.current_epoch != 0 and (self.current_epoch % self.update_loss_weights_per_epoch == 0):
                print('[!] Updating loss weights')
                self.loss_weights = self.softadapt.get_component_weights(torch.tensor(self.image_criterion_history),
                                                                         torch.tensor(
                                                                             self.audio_criterion_history),
                                                                         torch.tensor(
                                                                             self.fusion_criterion_history),
                                                                         verbose=True)
                print(f'[!] loss weights: {self.loss_weights}')
                self.image_criterion_history = list()
                self.audio_criterion_history = list()
                self.fusion_criterion_history = list()
        self.validation_step_outputs.clear()

    def setup_criterion(self) -> torch.nn.Module:
        return None

    def setup_scores(self) -> List[torch.nn.Module]:
        train_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                            f1m=F1Score(task="multiclass",
                                        num_classes=10, average='macro'),
                            prec_m=Precision(
                                task="multiclass", num_classes=10, average='macro'),
                            rec_m=Recall(task="multiclass", num_classes=10, average='macro'))
        val_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                          f1m=F1Score(task="multiclass",
                                      num_classes=10, average='macro'),
                          prec_m=Precision(task="multiclass",
                                           num_classes=10, average='macro'),
                          rec_m=Recall(task="multiclass", num_classes=10, average='macro'))
        test_scores = dict(acc=Accuracy(task="multiclass", num_classes=10),
                           f1m=F1Score(task="multiclass",
                                       num_classes=10, average='macro'),
                           prec_m=Precision(task="multiclass",
                                            num_classes=10, average='macro'),
                           rec_m=Recall(task="multiclass", num_classes=10, average='macro'))

        return [train_scores, val_scores, test_scores]

    def on_test_epoch_end(self, save_preds=False):
        outputs = self.test_step_outputs
        super().on_test_epoch_end(save_preds)
        preds = torch.cat([x['preds'] for x in outputs])
        preds_image = torch.cat([x['preds_image'] for x in outputs])
        preds_audio = torch.cat([x['preds_audio'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        image_logits = torch.cat([x['image_logits'] for x in outputs])
        audio_logits = torch.cat([x['audio_logits'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])
        loss_image = torch.stack([x['loss_image'] for x in outputs]).mean().item()
        loss_audio = torch.stack([x['loss_audio'] for x in outputs]).mean().item()
        loss_fusion = torch.stack([x['loss_fusion'] for x in outputs]).mean().item()
        self.image_criterion_history.append(loss_image)
        self.audio_criterion_history.append(loss_audio)
        self.fusion_criterion_history.append(loss_fusion)
        self.metrics_logger.log_metric("preds", preds)
        self.metrics_logger.log_metric("preds_image", preds_image)
        self.metrics_logger.log_metric("preds_audio", preds_audio)
        self.metrics_logger.log_metric("labels", labels)
        self.metrics_logger.log_metric("image_embeddings", image_logits)
        self.metrics_logger.log_metric("audio_embeddings", audio_logits)
        self.metrics_logger.log_metric("logits", logits)
        self.metrics_logger.log_metric("loss_image", self.image_criterion_history[-1])
        self.metrics_logger.log_metric("loss_audio", self.audio_criterion_history[-1])
        self.metrics_logger.log_metric("loss_fusion", self.fusion_criterion_history[-1])

     
        aleatoric_uncertainty = torch.cat(
            [x['aleatoric_uncertainty'] for x in outputs])
        epidemic_uncertainty = torch.cat(
            [x['epidemic_uncertainty'] for x in outputs])
        image_epidemic_uncertainty = torch.cat(
            [x['image_epidemic_uncertainty'] for x in outputs])
        image_aleatoric_uncertainty = torch.cat(
            [x['image_aleatoric_uncertainty'] for x in outputs])
        audio_epidemic_uncertainty = torch.cat(
            [x['audio_epidemic_uncertainty'] for x in outputs])
        audio_aleatoric_uncertainty = torch.cat(
            [x['audio_aleatoric_uncertainty'] for x in outputs])

        self.metrics_logger.log_metric("aleatoric_uncertainty", aleatoric_uncertainty)
        self.metrics_logger.log_metric("epidemic_uncertainty", epidemic_uncertainty)
        self.metrics_logger.log_metric("image_epidemic_uncertainty", image_epidemic_uncertainty)
        self.metrics_logger.log_metric("image_aleatoric_uncertainty", image_aleatoric_uncertainty)
        self.metrics_logger.log_metric("audio_epidemic_uncertainty", audio_epidemic_uncertainty)
        self.metrics_logger.log_metric("audio_aleatoric_uncertainty", audio_aleatoric_uncertainty)
        self.metrics_logger.save_metrics(os.path.join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}", "metrics.pt"))
        print("Metrics saved at", os.path.join(self.logger.save_dir, self.logger.name, f"version_{self.logger.version}", "metrics.pt"))
        if self.checkpoint_path is None:
            self.checkpoint_path = f'{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/checkpoints/'
        save_path = path.dirname(self.checkpoint_path)
        torch.save(dict(preds=preds, preds_image=preds_image, preds_audio=preds_audio, labels=labels,
                        image_logits=image_logits, audio_logits=audio_logits, logits=logits, aleatoric_uncertainty=aleatoric_uncertainty, epidemic_uncertainty=epidemic_uncertainty),
                   save_path + '/test_preds.pt')
        print(f'[!] Saved test predictions to {save_path}/test_preds.pt')
        self.test_step_outputs.clear()

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location=None,
            hparams_file: Optional = None,
            strict: bool = True,
            **kwargs: Any,
    ):
        model = super().load_from_checkpoint(checkpoint_path,
                                             map_location, hparams_file, strict, **kwargs)
        model.checkpoint_path = checkpoint_path
        return model

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), **optimizer_cfg)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=self.scheduler_patience)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def intermediate_step(self, batch: Any) -> Any:
        image = batch['image']
        audio = batch['audio']
        labels = batch['label']

        # get modality encodings from feature extractors
        image_logits = self.image_mixer(image)
        audio_logits = self.audio_mixer(audio)

        fused_moalities = self.fusion_function(image_logits, audio_logits)
        logits = self.fusion_mixer(fused_moalities)

        # fuse modalities
        results = self.shared_step(batch)

        fusion_correct = results['preds'] == labels
        image_correct = results['preds_image'] == labels
        audio_correct = results['preds_audio'] == labels

        return dict(image_logits=image_logits, audio_logits=audio_logits, logits=logits,
                    fusion_correct=fusion_correct, image_correct=image_correct, audio_correct=audio_correct)
