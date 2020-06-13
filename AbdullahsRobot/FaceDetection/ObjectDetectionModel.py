"""Object Detection Model 
    @author : Muhammad Sakti Alvissalim (alvissalim@gmail.com)

    Copyright (C) 2020  Muhammad Sakti Alvissalim 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import torch
from torch import nn
import torchvision.models as models

class ObjectDetectionModel(nn.Module):
    """ An object detection model for PyTorch """   
    def __init__(self, n_classes: int = 1, output_grid_size : int = 7, boxes_per_cell : int = 4, **kwarg):
        super(ObjectDetectionModel, self).__init__(**kwarg)
        self.n_classes = n_classes
        self.output_grid_size = output_grid_size
        self.boxes_per_cell = boxes_per_cell
        self.n_cells = output_grid_size * output_grid_size

        boxes_tensor_size = (boxes_per_cell * 5) # objectness score + (x,y,w,h)
        conditional_class_prob_tensor_size = n_classes

        cell_tensor_size = boxes_tensor_size + conditional_class_prob_tensor_size
        self.cell_tensor_size = cell_tensor_size
        base_model = models.mobilenet_v2()

        self.features_length = base_model.last_channel
        self.base_features = base_model.features
        
        self.final_feat = nn.Conv2d(base_model.last_channel, cell_tensor_size, 1, stride=1)

    def forward(self, x):
        x = self.base_features(x)
        x = self.final_feat(x)
        x = nn.functional.adaptive_avg_pool2d(x, self.output_grid_size )#
        x = x.permute(0,2,3,1)
        x = x.flatten(start_dim=1)

        return x

if __name__ == "__main__":
    dummy_input = torch.zeros(10,3,320,320)
    model = ObjectDetectionModel()
    x = model(dummy_input)
    print(x.size())