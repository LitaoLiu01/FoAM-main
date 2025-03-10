# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build_mt_act as build_mt_act
from .detr_vae import build_dream as build_dream
from .detr_vae import build_g_img_act as build_g_img_act
from .detr_vae import build_baku as build_baku
from .detr_vae import build_dream_wo_mph as build_dream_wo_mph
from .detr_vae import build_rt1 as build_rt1


def build_DREAM_model(args):
    return build_dream(args)

def build_DREAM_wo_MPH_model(args):
    return build_dream_wo_mph(args)

def build_MT_ACT_model(args):
    return build_mt_act(args)

def build_G_img_ACT_model(args):
    return build_g_img_act(args)

def build_baku_model(args):
    return build_baku(args)

def build_rt1_model(args):
    return build_rt1(args)

