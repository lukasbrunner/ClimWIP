#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright 2019 Lukas Brunner, ETH Zurich

This file is part of ClimWIP.

ClimWIP is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Authors
-------
- Lukas Brunner || lukas.brunner@env.ethz.ch

Abstract
--------
A collection of general utility functions.
"""
import os
import logging
import traceback
import numpy as np
import __main__ as main
from datetime import datetime
from configparser import ConfigParser
from munch import munchify

logger = logging.getLogger(__name__)
format_ = '%(asctime)s - %(levelname)s - %(funcName)s() %(lineno)s: %(message)s'


def set_logger(level=20, filename=None, format_=format_, **kwargs):
    """Set up a basic logger"""
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logging.basicConfig(
        level=level,
        filename=filename,
        format=format_,
        **kwargs)


def log_parser(args):
    def _print(ll, max_len=10):
        if isinstance(ll, (list, np.ndarray)):
            if len(ll) < max_len:
                return ', '.join(map(str, ll))
            else:
                return '{}, {}, ..., {}, {} (n={})'.format(*ll[:2], *ll[-2:], len(ll))
        return ll

    if type(args).__module__ == 'munch':
        logmsg = 'Read configuration file:\n\n'
        for ii in sorted(args.keys()):
            logmsg += '  {}: {}\n'.format(ii, _print(args[ii]))
    else:
        logmsg = 'Read parser input: \n\n'
        for ii, jj in sorted(vars(args).items()):
            logmsg += '  {}: {}\n'.format(ii, _print(jj))
    logger.info(logmsg)


def read_config(cname='DEFAULT', path='config.ini', separator=','):
    """Reads a config.ini file and returns a object containing
    its values.

    Parameters:
    - cname=None (str, optional): Name of the configuration to use.
    - path=None (str, optional): Path of the configuration file. Defaults to
      ./config.ini of the main file.
    - separator=',' (str, optional): Strings containing separator will be
      split into list. Set to None to disable this behavior.

    Returns:
    object

    The returned object contains all values as attributes plus one additional
    attribute config which contains 'cname'.

    Additional information:
    The ConfigParser method only returns strings by default. For convenience
    numerical expressions with a dot (e.g., 0.) will be converted to float and
    numerical expressions without a dot (e.g., 0) will be converted to int.
    To force a string of a number use explicit quotes (e.g., "0" or '0') in
    the .ini file. The strings 'False', 'True', and 'None' will be converted
    to their logical equivalents.  Strings containing the 'separator' string
    will be converted to list."""

    def _convert(str_):
        str_ = str_.strip()
        try:
            return int(str_)
        except ValueError:
            try:
                return float(str_)
            except ValueError:
                if str_ == 'True':
                    return True
                elif str_ == 'False':
                    return False
                elif str_ == 'None':
                    return None
                elif len(str_) == 0:  # NOTE: ignore empty strings in list
                    raise ValueError
                else:  # use '123' to force a string
                    return str(str_.replace("'", "").replace('"', ''))

    if not path.endswith('.ini'):
        path += '.ini'
    basepath = os.path.dirname(os.path.realpath(main.__file__))
    fullpath = os.path.join(basepath, path)

    if not os.path.isfile(fullpath):
        raise IOError('{} is not a valid filename'.format(fullpath))

    config = ConfigParser()
    config.read(fullpath)
    cc = munchify(dict(config[cname]))
    for key in cc.keys():
        if separator is not None and separator in cc[key]:
            cc[key] = cc[key].split(separator)
            for idx, _ in enumerate(cc[key]):
                try:
                    cc[key][idx] = _convert(cc[key][idx])
                except ValueError:  # NOTE: ignore empty strings in list
                    del cc[key][idx]
        else:
            cc[key] = _convert(cc[key])

    cc.config = cname  # also contain the name of the configuration
    cc.config_path = fullpath  # and path of the config file
    return cc


class LogTime:
    """A logger for keeping track of code timing.

    If called by 'with' log 'msg' with given 'level' before the intended code
    is executed. Log 'msg' again afterwards and add the status ('DONE' or
    'FAIL') as well as the execution time.

    The logger can be manually started and stopped by calling self.start() and
    self.stop (see examples).

    Parameters
    ----------
    msg : str, optional
        The default message to log.
    level : {'debug', 'info', 'warning'}, optional
        The default logging level.

    Examples
    --------
    with LogRegion('code description', level='info'):
        pass
    >>> INFO:code description...
    >>> INFO:code description... DONE (duration: 00:00:00.000000)

    with LogRegion('code description', level='info'):
        raise ValueError('error')
    >>> INFO:code description...
    >>> ERROR:code description... FAIL (duration: 00:00:00.000000)
    >>> ERROR:<Traceback>

    log = LogRegion('default message')
    log.start(level='debug')
    # calling start on a running logger will end the previous logger first
    log.start('other message')  # level will fall back to default
    log.stop
    >>> DEBUG:default message...
    >>> DEBUG:default message... DONE (duration: 00:00:00.000000)
    >>> INFO:other message...
    >>> INFO:other message... DONE (duration: 00:00:00.000000)

    with LogRegion('default message') as log:
        log.stop  # explicitly stop previous logger (optional)
        # piece of code which is not timed here
        log.start('other message', level='debug')
        log.start()  # fall back to defaults
        raise ValueError('error message')
    >>> INFO:default message...
    >>> INFO:default message... DONE (duration: 00:00:00.000000)
    >>> DEBUG:other message...
    >>> DEBUG:other message... DONE (duration: 00:00:00.000000)
    >>> INFO:default message...
    >>> ERROR:default message... FAIL (duration: 00:00:00.000000)
    >>> ERROR:<Traceback>
    """

    def __init__(self, msg='Start logging', level='info'):
        self.default_msg = msg
        self.default_level = level
        self.running = False

    def __enter__(self):
        self.start(self.default_msg, self.default_level)
        return self

    def __exit__(self, exception_type, exception_value, tb):
        if exception_type is None:
            self.stop
        else:
            self.level = 'error'
            self.stop
            # self.log_region(f'{exception_type.__name__}: {exception_value}')
            self.log_region(''.join(traceback.format_exception(
                exception_type, exception_value, tb)))

    def start(self, msg=None, level=None):
        """Log msg with given level.

        If LogTime is already running (because self.start() has already been
        called without a subsequent self.stop) this will also call self.stop
        before any other action (see self.stop for more information)

        Parameters
        ----------
        msg : str, optional
            Message to log.
        level : {'debug', 'info', 'warning'}, optional
            Overwrite class logging level for this call.
        """
        if self.running:
            self.stop
        self.running = True

        if msg is None:
            self.msg = self.default_msg
        else:
            self.msg = msg

        if level is None:
            self.level = self.default_level
        else:
            self.level = level

        self.t0 = datetime.now()
        self.log_region(f'{self.msg}...')

    @property
    def stop(self):
        """Log msg (from self.start) again and indicate time passed."""
        try:
            dt = datetime.now() - self.t0
        except AttributeError:
            raise ValueError('Timer not running, call self.start() first')
        if self.level == 'error':
            self.log_region(f'{self.msg}... FAIL (duration: {dt})')
        else:
            self.log_region(f'{self.msg}... DONE (duration: {dt})')
        self.running = False

    def log_region(self, msg):
        # this only exists to that the logger prints a nice function name
        self._logger(self.level)(msg)

    @staticmethod
    def _logger(level):
        """Set logging level from string"""
        if level.lower() == 'debug':
            return logger.debug
        elif level.lower() == 'info':
            return logger.info
        elif level.lower() == 'warning':
            return logger.warning
        elif level.lower() == 'error':
            return logger.error
