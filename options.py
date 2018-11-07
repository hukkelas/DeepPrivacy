from optparse import OptionParser
from wordgenerator import generate_random_word
import os
import math
DEFAULT_NUM_EPOCHS = 500
DEFAULT_BATCH_SIZE = 64
DEFAULT_N_CRITIC = 1
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NOISE_DIM = 256
DEFAULT_IMSIZE = 4
DEFAULT_MAX_IMSIZE = 256
DEFAULT_START_CHANNEL_SIZE = 256
DEFALUT_DATASET = "celeba"

# 4 -> 8 // 128 -> 64
# 8 -> 16 // 64 -> 32
# 16 -> 32 // 32 -> 16
# 

def validate_start_channel_size(max_imsize, start_channel_size):
    # Assert start channel size is valid with the max imsize
    # Number of times to double 
    n_image_double =  math.log(max_imsize, 2) - 2 # starts at 4
    n_channel_halving = math.log(start_channel_size, 2)
    assert n_image_double < n_channel_halving

def print_options(options):
    dic = vars(options)
    print("="*80)
    print("OPTIONS USED:")
    for (key, item) in dic.items():
        print("{:<16} {}".format(key, item))
    print("="*80)

def load_options():
    parser = OptionParser()
    parser.add_option("-b", "--batch-size", dest="batch_size",
                      help="Set batch size for training",
                      default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_option("-c", "--n-critic", dest="n_critic",
                      help="Set number of critic(discriminator) batch step per generator step",
                      default=DEFAULT_N_CRITIC, type=int)
    parser.add_option("-l", "--learning-rate", dest="learning_rate",
                      help="Set learning rate",
                      default=DEFAULT_LEARNING_RATE, type=float)
    parser.add_option("-z", "--noise-dim", dest="noise_dim",
                      help="Set dimension of noise data.",
                      default=DEFAULT_NOISE_DIM, type=int)
    parser.add_option("-e", "--num-epochs", dest="num_epochs",
                      help="Set number of epochs",
                      default=DEFAULT_NUM_EPOCHS, type=int)
    parser.add_option("--name", "--model-name", dest="model_name",
                      help="Set the name of the model",
                      default=generate_random_word(), type=str)
    parser.add_option("--imsize", dest="imsize",
                      help="Set the image size for discriminator and generator",
                      default=DEFAULT_IMSIZE, type=int)
    parser.add_option("--max-imsize", dest="max_imsize",
                      help="Set the final image size for the discriminator and generator",
                      default=DEFAULT_MAX_IMSIZE, type=int)
    parser.add_option("--start-channel-size", dest="start_channel_size",
                      help="Set the channel start size for Discriminator and Generator",
                      default=DEFAULT_START_CHANNEL_SIZE, type=int)
    parser.add_option("--dataset", dest="dataset",
                      help="Set the dataset to load",
                      default=DEFALUT_DATASET)

            
    options, _ = parser.parse_args()

    
    options.label_size = 10
    if options.dataset == "celeba":
        options.label_size = 0

    validate_start_channel_size(options.max_imsize, options.start_channel_size)


    options.checkpoint_dir = os.path.join("checkpoints", options.model_name)
    options.generated_data_dir = os.path.join("generated_data", options.model_name)
    options.summaries_dir = os.path.join("summaries", options.model_name)
    if os.path.isdir(options.summaries_dir):
        num_folders = len(os.listdir(options.summaries_dir))
        options.summaries_dir = os.path.join(options.summaries_dir, str(num_folders//3))
    else:
        options.summaries_dir = os.path.join(options.summaries_dir, str(0))
    os.makedirs(options.checkpoint_dir, exist_ok=True)
    os.makedirs(options.generated_data_dir, exist_ok=True)
    
    if options.dataset == "mnist":
        options.image_channels = 1
    else:
        options.image_channels = 3

    print_options(options)
    return options


if __name__ == '__main__':
    load_options()

    