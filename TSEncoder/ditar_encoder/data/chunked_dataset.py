"""Batch collation for the chunked latent-action datasets (HDF5 sources).

Each dataset returns a per-episode dict (``videos``, ``pad_mask``, ``action``,
``task_instruction``, and optionally a backfilled ``text_embedding`` /
``sampled_chunk_stride``). ``ChunkedCollatorForLatentAction`` stacks those into a
batch. Instruction text is normalized here so the train-time T5 conditioning
matches inference — see :mod:`ditar_encoder.encoders.text_normalization`.
"""
import torch

from ditar_encoder.encoders.text_normalization import normalize_instruction


class ChunkedCollatorForLatentAction:
    def __call__(self, instances):
        output = dict(
            videos=torch.stack([instance['videos'] for instance in instances], dim=0),
            pad_mask=torch.stack([instance['pad_mask'] for instance in instances], dim=0),
            action=torch.stack([instance['action'] for instance in instances], dim=0),
            task_instruction=[
                normalize_instruction(instance['task_instruction'])
                for instance in instances
            ],
        )
        has_text_embedding = ['text_embedding' in instance for instance in instances]
        if any(has_text_embedding) and not all(has_text_embedding):
            raise ValueError(
                'Mixed batches with partially backfilled text_embedding are not supported. '
                'Please finish backfilling all HDF5 files.'
            )
        if all(has_text_embedding):
            output['text_embedding'] = torch.stack(
                [instance['text_embedding'] for instance in instances], dim=0
            )
        if 'dataset_name' in instances[0]:
            output['dataset_names'] = [instance['dataset_name'] for instance in instances]
        if 'sampled_chunk_stride' in instances[0]:
            output['sampled_chunk_stride'] = torch.stack(
                [instance['sampled_chunk_stride'] for instance in instances], dim=0
            )
        if 'sampled_gap_strides' in instances[0]:
            output['sampled_gap_strides'] = torch.stack(
                [instance['sampled_gap_strides'] for instance in instances], dim=0
            )
        if 'eval_loader_tag' in instances[0]:
            output['eval_loader_tag'] = instances[0]['eval_loader_tag']
        return output
