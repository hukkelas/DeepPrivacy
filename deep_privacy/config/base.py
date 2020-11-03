import pathlib
import tempfile
import shutil
import sys
import json
from importlib import import_module
from addict import Dict
from deep_privacy import logger


def isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError("'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name))
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class Config(object):
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"

    """

    @staticmethod
    def _py2dict(filepath: pathlib.Path):

        assert filepath.is_file(), filepath
        with tempfile.TemporaryDirectory() as temp_config_dir:
            shutil.copyfile(filepath,
                            pathlib.Path(temp_config_dir, '_tempconfig.py'))
            sys.path.insert(0, temp_config_dir)
            mod = import_module('_tempconfig')
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__') and name != "os"
            }
            # delete imported module
            del sys.modules['_tempconfig']

        cfg_text = f"{filepath}\n"
        with open(filepath, 'r') as f:
            cfg_text += f.read()
        if '_base_config_' in cfg_dict:
            cfg_dir = filepath.parent
            base_filename = cfg_dict.pop('_base_config_')
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(
                    cfg_dir.joinpath(f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError('Duplicate key is not allowed among bases')
                base_cfg_dict.update(c)

            Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)
        return cfg_dict, cfg_text

    @staticmethod
    def _json2dict(filepath: pathlib.Path):
        with open(filepath, "r") as fp:
            return json.load(fp)

    @staticmethod
    def _file2dict(filepath):
        filepath = pathlib.Path(filepath)
        if filepath.suffix == ".py":
            return Config._py2dict(filepath)
        if filepath.suffix == ".json":
            return Config._json2dict(filepath), None
        raise ValueError("Expected json or python file:", filepath)

    @staticmethod
    def _merge_a_into_b(a, b):
        # merge dict a into dict b. values in a will overwrite b.
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                if not isinstance(b[k], dict):
                    raise TypeError(
                        'Cannot inherit key {} from base!'.format(k))
                Config._merge_a_into_b(v, b[k])
            else:
                if not k in b:
                    logger.warn(
                        f"Writing a key without a default value: key={k}")
                b[k] = v

    @staticmethod
    def fromfile(filepath):
        cfg_dict, cfg_text = Config._file2dict(filepath)
        return Config(cfg_dict, cfg_text=cfg_text, filename=filepath)

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but got {}'.format(
                type(cfg_dict)))

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def model_name(self):
        return pathlib.Path(self._filename).stem

    @property
    def text(self):
        return self._text

    @property
    def output_dir(self):
        parts = pathlib.Path(self.filename).parts
        parts = [
            pathlib.Path(p).stem for p in parts
            if "configs" not in p]
        return pathlib.Path(self._output_dir, *parts)

    @property
    def cache_dir(self):
        return pathlib.Path(self._cache_dir)

    def __repr__(self):
        cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
        return json.dumps(cfg_dict, indent=2)
        return 'Config (path: {}): {}'.format(self.filename,
                                              self._cfg_dict.__repr__())

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def dump(self):
        filepath = self.output_dir.joinpath("config_dump.json")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
        with open(filepath, "w") as fp:
            json.dump(cfg_dict, fp, indent=4)

    def merge_from_dict(self, options):
        """ Merge list into cfg_dict

        Merge the dict parsed by MultipleKVAction into this cfg.
        Example,
            >>> options = {'model.backbone.depth': 50}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)

        Args:
            options (dict): dict of configs to merge from.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d[subkey] = ConfigDict()
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
        Config._merge_a_into_b(option_cfg_dict, cfg_dict)

    def merge_from_str(self, opts: str):
        if opts is None:
            return
        b = {}
        for opt in opts.split(","):
            try:
                key, value = opt.split(" ")
            except ValueError:
                key, value = opt.split("=")
            if isfloat(value):
                value = float(value)
            b[key] = value
        self.merge_from_dict(b)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    cfg = Config.fromfile(parser.parse_args().filepath)
    print("Ouput directory", cfg.output_dir)
    cfg.dump()
