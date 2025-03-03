import torch

from hallucination.model.trfold.embedding import TR_embedding
from hallucination.model.trfold.msa import msa_parser
from hallucination.model.trfold.trfold_net import TRFold


def trfold_predict(model, msa, template, idx, aatype, recycles, bf16=True):
    model.eval()
    template = torch.unsqueeze(template, dim=0)
    idx = torch.unsqueeze(idx, dim=0)
    aatype = torch.unsqueeze(aatype, dim=0)

    if bf16:
        template = template.bfloat16()

    L = idx.shape[0]
    if torch.cuda.is_available():
        template = template.cuda()
        idx = idx.cuda()
        aatype = aatype.cuda()
    with torch.no_grad():
        if torch.cuda.is_available():
            model.recycle_query = torch.zeros((1, L, 256), device="cuda:0")
            model.recycle_rr = torch.zeros((1, L, L, 128), device="cuda:0")
            model.recycle_dist = torch.zeros((1, L, L, 15), device="cuda:0")
        else:
            model.recycle_query = torch.zeros((1, L, 256), device="cpu")
            model.recycle_rr = torch.zeros((1, L, L, 128), device="cpu")
            model.recycle_dist = torch.zeros((1, L, L, 15), device="cpu")
        if bf16:
            model.recycle_query = model.recycle_query.bfloat16()
            model.recycle_rr = model.recycle_rr.bfloat16()
            model.recycle_dist = model.recycle_dist.bfloat16()
        for _ in range(recycles):
            input_msa, extra_msa, _ = msa_parser(msa.clone(), 256, 2048, bf16)
            input_msa = input_msa.unsqueeze(0)
            extra_msa = extra_msa.unsqueeze(0)
            if torch.cuda.is_available():
                input_msa = input_msa.cuda()
                extra_msa = extra_msa.cuda()
            model(input_msa, extra_msa, template, idx, aatype)
        input_msa, extra_msa, _ = msa_parser(msa.clone(), 256, 2048, bf16)
        input_msa = input_msa.unsqueeze(0)
        extra_msa = extra_msa.unsqueeze(0)
        if torch.cuda.is_available():
            input_msa = input_msa.cuda()
            extra_msa = extra_msa.cuda()
        res = model(input_msa, extra_msa, template, idx, aatype)
    return res


def logits_to_probs(
    logits, add_gumbel_noise=False, output_type="hard", temp=1, eps=1e-8
):
    device = logits.device
    if add_gumbel_noise:
        U = torch.rand(logits.shape)
        noise = -torch.log(-torch.log(U + eps) + eps)
        noise = noise.to(device)
        logits = logits + noise
    y_soft = torch.nn.functional.softmax(logits / temp, -1)
    if output_type == "soft":
        msa = y_soft
    elif output_type == "hard":
        n_cat = y_soft.shape[-1]
        y_oh = torch.nn.functional.one_hot(y_soft.argmax(-1), n_cat)
        y_hard = (y_oh - y_soft).detach() + y_soft
        msa = y_hard
    else:
        raise AssertionError(
            f'output type should be "soft" or "hard", got {output_type}'
        )
    return msa


def load_struture_module(model_path, device="cuda"):
    model = TRFold(
        blocks=48,
        MSA_channels=256,
        pair_channels=128,
        device=device,
        embedding_model_type=TR_embedding.EMBEDDING_MODEL_TYPE_48BLOCK,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model
