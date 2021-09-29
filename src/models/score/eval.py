import types
import torch
import numpy as np

from tqdm import tqdm

from .inception import InceptionV3
from .fid import calculate_frechet_distance, torch_cov

from torchmetrics import FID, IS

device = torch.device('cuda:0')

fid_module = FID(feature=2048)
is_module = IS(feature=2048)

def get_inception_and_fid_score(
        images,
        num_images=None,
        splits=10,
        batch_size=50,
        verbose=False
):
    """when `images` is a python generator, `num_images` should be given"""

    if num_images is None and isinstance(images, types.GeneratorType):
        raise ValueError(
            "when `images` is a python generator, "
            "`num_images` should be given")

    if num_images is None:
        num_images = len(images)

    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    block_idx2 = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx1, block_idx2]).to(device)
    model.eval()

    fid_acts = torch.empty((num_images, 2048)).to(device)
    is_probs = torch.empty((num_images, 1008)).to(device)

    iterator = iter(tqdm(
        images, total=num_images,
        dynamic_ncols=True, leave=False, disable=not verbose,
        desc="get_inception_and_fid_score")
    )

    start = 0
    while True:
        batch_images = []
        # get a batch of images from iterator
        try:
            for _ in range(batch_size):
                batch_images.append(next(iterator))
        except StopIteration:
            if len(batch_images) == 0:
                break
            pass
        batch_images = np.stack(batch_images, axis=0)
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
            fid_acts[start: end] = pred[0].view(-1, 2048)
            is_probs[start: end] = pred[1]
        start = end

    # Inception Score
    scores = []
    for i in range(splits):
        part = is_probs[
               (i * is_probs.shape[0] // splits):
               ((i + 1) * is_probs.shape[0] // splits), :]
        kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
        kl = torch.mean(torch.sum(kl, 1))
        scores.append(torch.exp(kl))

    scores = torch.stack(scores)
    is_score = (torch.mean(scores).cpu().item(),
                torch.std(scores).cpu().item())

    # FID Score
    f = np.load(fid_cache)
    m2, s2 = f['mu'][:], f['sigma'][:]
    f.close()

    m1 = torch.mean(fid_acts, axis=0)
    s1 = torch_cov(fid_acts, rowvar=False)
    m2 = torch.tensor(m2).to(m1.dtype).to(device)
    s2 = torch.tensor(s2).to(s1.dtype).to(device)

    fid_score = calculate_frechet_distance(m1, s1, m2, s2)

    del fid_acts, is_probs, scores, model
    return is_score, fid_score


def evaluate(sampler, fid_cache: str, sample_size: int, img_size: int):
    with torch.no_grad():
        images = []
        for i in range(0, sample_size, 32):
            batch_size = min(32, sample_size - i)
            x_T = torch.randn((batch_size, 3, img_size, img_size))

            # sample batch image
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)

        images = torch.cat(images, dim=0).numpy()

    (IS, IS_std), FID = get_inception_and_fid_score(
        images, fid_cache=fid_cache, num_images=sample_size, verbose=True
    )
    return (IS, IS_std), FID, images
