#!/usr/bin/python2.7

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
from eval import segment_bars_with_confidence
from  cnn_simplified import GCNnet_artisticswimming_simplified as GCN
from  cnn_simplified2 import  GCNnet_artisticswimming_simplified as GOAT_GCN
from group_aware_attention  import Encoder_Blocks as Mutihead_cross
from attention import  Encoder_Blocks as Mutihead
from  datasets import  VideoDataset
import Gconfig
class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out
    
class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out
class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes,args):

        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        if args.goat==0:
            self.mh = Mutihead(args.num_features_gcn, dim, args.linea_dim, 8, args.num_layers, attn_drop=args.attn_drop)
            self.gcn = GCN(args)
        else:
            self.mh = Mutihead_cross(args.num_features_gcn, dim, args.linea_dim, 8, args.num_layers, attn_drop=args.attn_drop)
            self.gcn = GOAT_GCN(args)
        self.args=args
    def setgoat(self,file,ft):
        a = self.args
        self.tdataset = VideoDataset(mode="train",args=a,files=file)
        self.testdataset = VideoDataset(mode="test",args=a,files=ft)
    def trainG(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.gcn.train()
        self.mh.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.gcn.to(device)
        self.mh.to(device)
        optimizer = optim.Adam([{'params': self.model.parameters(),'lr':learning_rate},
                                {'params': self.gcn.parameters(),'lr':self.args.goat_lr},{'params': self.mh.parameters(),'lr':self.args.goat_lr}])

        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():

                batch_input, batch_target, mask, vids = batch_gen.next_batch(batch_size)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if batch_input.size(2) >= 7050:
                    continue
                optimizer.zero_grad()
                with torch.no_grad():
                    if batch_size > 1:

                        boxes_features=[]
                        boxes_in=[]
                        for v in vids:
                            ts = vids[0].split('_')
                            st = [s + '_' for s in ts]
                            sts = [st[i] for i in range(len(st) - 1)]
                            sss = '_'
                            for s in sts:
                                sss += s
                            sss =sss.strip('_')
                            data = self.tdataset.next_batch(batch_size,(sss,int(ts[-1])))
                            boxes_features.append(data['cnn_features'])
                            boxes_in.append( data['boxes'])  # B,T,N,4
                        boxes_in_1=torch.stack(boxes_in).to(device)
                        boxes_features_1=torch.stack(boxes_features).to(device)
                        batch_input =batch_input.permute(0,2,1)
                    else:
                        ts = vids[0].split('_')
                        st = [s + '_' for s in ts]
                        sts = [st[i] for i in range(len(st) - 1)]
                        sss = '_'
                        for s in sts:
                            sss += s
                        sss =sss.strip('_')
                        data = self.tdataset.next_batch(batch_size,(sss,int(ts[-1])))
                        boxes_features_1 = data['cnn_features'].to(device).unsqueeze(0)
                        boxes_in_1 = data['boxes'].to(device).unsqueeze(0)  # B,T,N,4
                        batch_input = batch_input.squeeze(0).T.unsqueeze(0)
                q1 = self.gcn(boxes_features_1, boxes_in_1)  # B,540,1024
                k1 = q1
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                if self.args.goat == 0:
                    batch_input = self.mh(q1, k1 , batch_input).to(device)
                else:
                    if batch_size>1:
                        print(batch_input.shape)
                        batch_input =self.mh(batch_input,q1,k1).to(device).squeeze(0).permute(2,0,1)
                        print(batch_input.shape)
                    else:
                        batch_input = self.mh(batch_input, q1, k1).to(device)
                predictions = self.model(batch_input)# B,D,T
                loss = 0
                if batch_size>1:
                    print(vids)
                    print(predictions.shape)
                    print(batch_input.shape)
                    print(batch_target)
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.mh.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".mh")
            torch.save(self.gcn.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".gch")
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            logger.info("split:  {}".format(self.args.split)+
                "[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                             float(correct) / total))

    def predictG(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate):
        self.model.eval()
        self.gcn.eval()
        self.mh.eval()
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.gcn.to(device)
            self.mh.to(device)
            print(epoch)
            print(model_dir)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"), strict=False)
            self.gcn.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".gch"), strict=False)
            self.mh.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".mh"), strict=False)
            batch_gen_tst.reset()
            while batch_gen_tst.has_next():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
                vid = vids[0] + '.mp4'
             
                ts = vids[0].split('_')
                st = [s + '_' for s in ts]
                sts = [st[i] for i in range(len(st) - 1)]
                sss = '_'
                for s in sts:
                    sss += s
                sss = sss.strip('_')
                input_x = torch.tensor(batch_input, dtype=torch.float)
                input_x.unsqueeze(0)
                input_x = input_x.to(device)
                data = self.tdataset.next_batch(1,(sss,int(ts[-1])))
                boxes_features_1 = data['cnn_features'].to(device).unsqueeze(0)
                boxes_in_1 = data['boxes'].to(device).unsqueeze(0) # B,T,N,4
                q1 = self.gcn(boxes_features_1, boxes_in_1)  # B,540,1024
                k1 = q1
                input_x = input_x.squeeze(0).T.unsqueeze(0)

                if self.args.goat == 0:
                    input_x = self.mh(q1, k1 , input_x).to(device)
                else:
                    input_x = self.mh(input_x,q1,k1).to(device) # B,540,1024 # B,1024
                predictions = self.model(input_x)

                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()

                    batch_target = batch_target.squeeze()
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()

                    segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
                                                 confidence.tolist(),
                                                 batch_target.tolist(), predicted.tolist())

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask,vids = batch_gen.next_batch(batch_size)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate):
        self.model.eval()
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"), strict=False)

            batch_gen_tst.reset()
            import time

            time_start = time.time()
            while batch_gen_tst.has_next():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
                vid = vids[0] + '.mp4'
             
                input_x = torch.tensor(batch_input, dtype=torch.float)
                input_x.unsqueeze(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)

                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()

                    batch_target = batch_target.squeeze()
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()

                    segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
                                                 confidence.tolist(),
                                                 batch_target.tolist(), predicted.tolist())

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
            time_end = time.time()
