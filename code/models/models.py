import torch.nn as nn
import torch


def normalization(x, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(x)
    elif norm == 'gn':
        m = nn.GroupNorm(4, x)
    elif norm == 'in':
        m = nn.InstanceNorm3d(x)
    return m


class Model(nn.Module):

    def __init__(self, num_cls=4, basic_dims=16):
        super(Model, self).__init__()
        self.num_cls = num_cls
        
        self.flair_encoder = Encoder(basic_dims)
        self.t1ce_encoder = Encoder(basic_dims)
        self.t1_encoder = Encoder(basic_dims)
        self.t2_encoder = Encoder(basic_dims)

        self.decoder_fuse = Decoder_fuse(num_cls, basic_dims)
        self.decoder_sep = Decoder_sep(num_cls, basic_dims)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) 

    def forward(self, x, mask):

        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])
            
        x1 = torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1) 
        x2 = torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1)
        x3 = torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1)
        x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)
    
        fuse_pred = self.decoder_fuse(x1, x2, x3, x4, mask)
        if self.training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred)
        return fuse_pred


class general_conv3d(nn.Module):

    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='reflect', norm='in', act_type='lrelu', relufactor=0.2):
        super(general_conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)
        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Encoder(nn.Module):

    def __init__(self, basic_dims):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d(1, basic_dims)
        self.e1_c2 = general_conv3d(basic_dims, basic_dims)
        self.e1_c3 = general_conv3d(basic_dims, basic_dims)

        self.e2_c1 = general_conv3d(basic_dims, basic_dims*2, stride=2)
        self.e2_c2 = general_conv3d(basic_dims*2, basic_dims*2)
        self.e2_c3 = general_conv3d(basic_dims*2, basic_dims*2)

        self.e3_c1 = general_conv3d(basic_dims*2, basic_dims*4, stride=2)
        self.e3_c2 = general_conv3d(basic_dims*4, basic_dims*4)
        self.e3_c3 = general_conv3d(basic_dims*4, basic_dims*4)

        self.e4_c1 = general_conv3d(basic_dims*4, basic_dims*8, stride=2)
        self.e4_c2 = general_conv3d(basic_dims*8, basic_dims*8)
        self.e4_c3 = general_conv3d(basic_dims*8, basic_dims*8)

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4


class Decoder_sep(nn.Module):

    def __init__(self, num_cls, basic_dims):
        super(Decoder_sep, self).__init__()

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4)
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4)
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, stride=1, padding=0)

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2)
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2)
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, stride=1, padding=0)

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims)
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims)
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, stride=1, padding=0)

        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))
        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))

        de_x3 = self.d2_c1(self.d2(de_x3))
        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        
        de_x2 = self.d1_c1(self.d1(de_x2))
        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)
        return pred


class Decoder_fuse(nn.Module):

    def __init__(self, num_cls, basic_dims):
        super(Decoder_fuse, self).__init__()

        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4)
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4)
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, stride=1, padding=0)

        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2)
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2)
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, stride=1, padding=0)

        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims)
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims)
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, stride=1, padding=0)

        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.FM4 = modal_fusion(in_channel=basic_dims*8)
        self.FM3 = modal_fusion(in_channel=basic_dims*4)
        self.FM2 = modal_fusion(in_channel=basic_dims*2)
        self.FM1 = modal_fusion(in_channel=basic_dims*1)


    def forward(self, x1, x2, x3, x4, mask):
  
        de_x4 = self.FM4(x4, mask)
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.FM3(x3, mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.FM2(x2, mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))
        
        de_x1 = self.FM1(x1, mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred


class avg_fusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, flair_x, t1ce_x, t1_x, t2_x, mask):
        out = flair_x + t1ce_x + t1_x + t2_x
        mask = torch.sum(mask, dim=1).reshape(mask.shape[0],1,1,1,1)
        out = out / mask
        return out
        

class modal_fusion(nn.Module):

    def __init__(self, in_channel):
        super(modal_fusion, self).__init__()
        self.fusion_layer = avg_fusion()
        self.conv_layer = nn.Sequential(
                            general_conv3d(in_channel, in_channel, k_size=1, padding=0, stride=1),
                            general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                            general_conv3d(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        
        out = self.fusion_layer(y[:,0,...], y[:,1,...], y[:,2,...], y[:,3,...], mask)
        out = self.conv_layer(out)
        return out

