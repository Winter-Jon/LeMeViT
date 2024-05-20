import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn



from timm.models.layers import trunc_normal_

from timm.models.vision_transformer import PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed
# from positional_encodings import PositionalEncodingPermute2D, Summer
# from siren_pytorch import SirenNet
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

from LeMeViT import LeMeViTBlock


class LeMeViT_Plain(nn.Module):
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 depth=12, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 head_dim=64, 
                 mlp_ratios=4, 
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop=0., 
                 drop_path_rate=0.,
                 # <<<------
                 attn_type="M",
                 queries_len=16,
                 qk_dims=None,
                 cpe_ks=0,
                 pre_norm=True,
                 mlp_dwconv=False,
                 representation_size=None,
                 layer_scale_init_value=-1,
                 use_checkpoint_stages=[],
                 # ------>>>
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        qk_dims = qk_dims or embed_dim
        
        # self.num_stages = len(attn_type)
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, flatten=True)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        ############ downsample layers (patch embeddings) ######################
        # self.downsample_layers = nn.ModuleList()
        # # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv 
        # stem = nn.Sequential(
        #     nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        #     nn.BatchNorm2d(embed_dim[0] // 2),
        #     nn.GELU(),
        #     nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        #     nn.BatchNorm2d(embed_dim[0]),
        # )

        # if use_checkpoint_stages:
        #     stem = checkpoint_wrapper(stem)
        # self.downsample_layers.append(stem)

        # for i in range(self.num_stages-1):
        #     if attn_type[i] == "STEM":
        #         downsample_layer = nn.Identity()
        #     else:
        #         downsample_layer = nn.Sequential(
        #             nn.Conv2d(embed_dim[i], embed_dim[i+1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        #             nn.BatchNorm2d(embed_dim[i+1])
        #         )
        #     if use_checkpoint_stages:
        #         downsample_layer = checkpoint_wrapper(downsample_layer)
        #     self.downsample_layers.append(downsample_layer)
        ##########################################################################


        attn_type = ["C", "C", "D2", "D2", "C"]
        attn_type.extend(["S" for i in range(depth-3)])

        #TODO: maybe remove last LN
        self.queries_len = queries_len
        self.meta_token = nn.Parameter(torch.randn(1,self.queries_len ,embed_dim), requires_grad=True) 
        
        nheads= qk_dims // head_dim
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        
        self.blocks = nn.ModuleList(
                [LeMeViTBlock(
                    dim=embed_dim, 
                    attn_drop=attn_drop, proj_drop=drop_rate,
                    drop_path=dp_rates[i],
                    attn_type=attn_type[i],
                    layer_scale_init_value=layer_scale_init_value,
                    num_heads=nheads,
                    qk_dim=qk_dims,
                    mlp_ratio=mlp_ratios,
                    mlp_dwconv=mlp_dwconv,
                    cpe_ks=cpe_ks,
                    pre_norm=pre_norm
                ) for i in range(depth)],
        )
        # self.prototype_downsample = nn.ModuleList()
        # prototype_downsample = nn.Sequential(
        #     nn.Linear(embed_dim[0], embed_dim[0] * 4),
        #     nn.LayerNorm(embed_dim[0] * 4),
        #     nn.GELU(),
        #     nn.Linear(embed_dim[0] * 4, embed_dim[0]),
        #     nn.LayerNorm(embed_dim[0])
        # )
        # self.prototype_downsample.append(prototype_downsample)
        # for i in range(self.num_stages-1):
        #     prototype_downsample = nn.Sequential(
        #         nn.Linear(embed_dim[i], embed_dim[i] * 4),
        #         nn.LayerNorm(embed_dim[i] * 4),
        #         nn.GELU(),
        #         nn.Linear(embed_dim[i] * 4, embed_dim[i+1]),
        #         nn.LayerNorm(embed_dim[i+1])
        #     )
        #     self.prototype_downsample.append(prototype_downsample)

        # self.prototype_stem = nn.ModuleList()
        # for i in range(4):
        #     prototype_stem = StemAttention(embed_dim[0], num_heads=2, bias=True)
        #     self.prototype_stem.append(prototype_stem)
        
        # self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks

        # cur = 0
        # for i in range(self.num_stages):
        #     stage = nn.ModuleList(
        #         [LeMeViTBlock(dim=embed_dim[i], 
        #                    attn_drop=attn_drop, proj_drop=drop_rate,
        #                    drop_path=dp_rates[cur + j],
        #                    attn_type=attn_type[i],
        #                    layer_scale_init_value=layer_scale_init_value,
        #                    num_heads=nheads[i],
        #                    qk_dim=qk_dims[i],
        #                    mlp_ratio=mlp_ratios[i],
        #                    mlp_dwconv=mlp_dwconv,
        #                    cpe_ks=cpe_ks,
        #                    pre_norm=pre_norm
        #             ) for j in range(depth[i])],
        #     )
        #     if i in use_checkpoint_stages:
        #         stage = checkpoint_wrapper(stage)
        #     self.stages.append(stage)
        #     cur += depth[i]

        ##########################################################################
        self.norm = nn.LayerNorm(embed_dim)
        self.norm_c = nn.LayerNorm(embed_dim)
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.meta_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, c):

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed
        
        for i, block in enumerate(self.blocks):
            x, c = block(x, c)
            
        x = self.norm(x)
        c = self.norm_c(c)
        
        x = self.pre_logits(x)
        c = self.pre_logits(c)

        # x = x.flatten(2).mean(-1,keepdim=True)
        # c = c.transpose(-2,-1).contiguous().mean(-1,keepdim=True)
        # x = torch.concat([x,c],dim=-1).mean(-1)

        x = x.flatten(2).mean(-1)
        c = c.transpose(-2,-1).contiguous().mean(-1)
        x = c

        return x

    def forward(self, x):
        B, _, H, W = x.shape 
        c = self.meta_token.repeat(B,1,1)
        x = self.forward_features(x, c)
        x = self.head(x)
        return x


def lemevit_base_patch16(**kwargs):
    model = LeMeViT_Plain(
        patch_size=16, embed_dim=768, depth=12, head_dim=64, mlp_ratios=4, qkv_bias=True,)
    return model


def lemevit_large_patch16(**kwargs):
    model = LeMeViT_Plain(
        patch_size=16, embed_dim=1024, depth=24, head_dim=64, mlp_ratios=4, qkv_bias=True,)
    return model


def lemevit_huge_patch14(**kwargs):
    model = LeMeViT_Plain(
        patch_size=14, embed_dim=1280, depth=32, head_dim=64, mlp_ratios=4, qkv_bias=True,)
    return model