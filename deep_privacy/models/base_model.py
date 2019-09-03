from torch import nn

class ProgressiveBaseModel(nn.Module):

    def __init__(self, pose_size, start_channel_size, image_channels):
        super().__init__()
        
        self.transition_channels = [
            start_channel_size,
            start_channel_size,
            start_channel_size,
            start_channel_size//2,
            start_channel_size//4,
            start_channel_size//8,
            start_channel_size//16,
            start_channel_size//32,
        ]
        self.transition_channels = [x // 8 * 8 for x in self.transition_channels]
        self.image_channels = image_channels
        self.transition_value = 1.0
        self.num_poses = pose_size // 2
        self.current_imsize = 4
        self.transition_step = 0
        self.prev_channel_extension = start_channel_size
    
    def extend(self):
        self.transition_value = 0.0
        self.prev_channel_extension = self.transition_channels[self.transition_step]
        self.transition_step += 1
        self.current_imsize *= 2
    

    def state_dict(self):
        return {
            "transition_step": self.transition_step,
            "transition_value": self.transition_value,
            "parameters": super().state_dict()
        }
    
    def load_state_dict(self, ckpt):
        for i in range(ckpt["transition_step"]):
            self.extend()
        self.transition_value = ckpt["transition_value"]

        super().load_state_dict(ckpt["parameters"])

