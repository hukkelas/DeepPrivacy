from optparse import OptionParser
from random_word_generator.main import random_word
import os
DEFAULT_NUM_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_N_CRITIC = 4
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_NOISE_DIM = 100


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
                      default=random_word(), type=str)
    options, _ = parser.parse_args()
    options.checkpoint_dir = os.path.join("checkpoints", options.model_name)
    options.generated_data_dir = os.path.join("generated_data", options.model_name)
    options.summaries_dir = os.path.join("summaries", options.model_name)
    os.makedirs(options.checkpoint_dir, exist_ok=True)
    os.makedirs(options.generated_data_dir, exist_ok=True)

    print_options(options)
    return options


if __name__ == '__main__':
    load_options()

    