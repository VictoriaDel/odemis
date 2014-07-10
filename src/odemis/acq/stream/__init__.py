# -*- coding: utf-8 -*-
'''
Created on 25 Jun 2014

@author: Rinze de Laat

Copyright © 2013-2014 Rinze de Laat, Éric Piel, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 2 as published by the Free Software Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Odemis. If not, see http://www.gnu.org/licenses/.
'''


# This module contains classes that describe Streams, which are basically
# Detector, Emitter and Dataflow associations.


from __future__ import division

from ._base import *
from ._helper import *
from ._live import *
from ._static import *
from ._sync import *

# Generic cross-cut types

# All the stream types related to optical
OPTICAL_STREAMS = (CameraStream,
                   StaticFluoStream,
                   StaticBrightfieldStream)

# All the stream types related to electron microscope
EM_STREAMS = (SEMStream,
              StaticSEMStream)

SPECTRUM_STREAMS = (SpectrumStream,
                    StaticSpectrumStream,
                    SEMSpectrumMDStream)

AR_STREAMS = (ARStream,
              StaticARStream,
              SEMARMDStream)

# TODO: make it like a VA, so that it's possible to know when it changes
class StreamTree(object):
    """ Object which contains a set of streams, and how they are merged to
    appear as one image. It's a tree which has one stream per leaf and one merge
    operation per node. => recursive structure (= A tree is just a node with
    a merge method and a list of subnodes, either streamtree as well, or stream)
    """

    def __init__(self, operator=None, streams=None, **kwargs):
        """
        :param operator: (callable) a function that takes a list of
            RGB DataArrays in the same order as the streams are given and the
            additional arguments and returns one DataArray.
            By default operator is an average function.
        :param streams: (list of Streams or StreamTree): a list of streams, or
            StreamTrees.
            If a StreamTree is provided, its outlook is first computed and then
            passed as an RGB DataArray.
        :param kwargs: any argument to be given to the operator function
        """
        self.operator = operator or img.Average

        streams = streams or []
        assert(isinstance(streams, list))

        self.streams = []
        self.should_update = model.BooleanVA(False)
        self.kwargs = kwargs

        for s in streams:
            self.add_stream(s)

    def __str__(self):
        return "[" + ", ".join([str(s) for s in self.streams]) + "]"

    def __len__(self):
        acc = 0

        for s in self.streams:
            if isinstance(s, Stream):
                acc += 1
            elif isinstance(s, StreamTree):
                acc += len(s)

        return acc

    def __getitem__(self, index):
        """ Return the Stream of StreamTree using index reference val[i] """
        return self.streams[index]

    def add_stream(self, stream):
        if isinstance(stream, (Stream, StreamTree)):
            self.streams.append(stream)
            if hasattr(stream, 'should_update'):
                stream.should_update.subscribe(self.stream_update_changed,
                                               init=True)
            # print "stream added %s" % stream.should_update.value
        else:
            msg = "Illegal type %s found in add_stream!" % type(stream)
            raise ValueError(msg)

    def remove_stream(self, stream):
        if hasattr(stream, 'should_update'):
            stream.should_update.unsubscribe(self.stream_update_changed)
        self.streams.remove(stream)
        self.stream_update_changed()

    def stream_update_changed(self, should_update=None):
        """ This method is called when one of the streams' should_update
        vigilant attribute changes.
        """
        # At least one stream is live, so we 'should update'
        for s in self.streams:
            if hasattr(s, "should_update") and s.should_update.value:
                self.should_update.value = True
                break
        else:
            self.should_update.value = False

    def getStreams(self):
        """
        Return the set of streams used to compose the picture. In other words,
        the leaves of the tree.
        """
        leaves = set()
        for s in self.streams:
            if isinstance(s, Stream):
                leaves.add(s)
            elif isinstance(s, StreamTree):
                leaves |= s.getStreams()

        return leaves

    def getImage(self, rect, mpp):
        """
        Returns an image composed of all the current stream images.
        Precisely, it returns the output of a call to operator.
        rect (2-tuple of 2-tuple of float): top-left and bottom-right points in
          world position (m) of the area to draw
        mpp (0<float): density (meter/pixel) of the image to compute
        """
        # TODO: probably not so useful function, need to see what canvas
        #  it will likely need as argument a wx.Bitmap, and view rectangle
        #  that will define where to save the result

        # TODO: cache with the given rect and mpp and last update time of each
        # image

        # create the arguments list for operator
        images = []
        for s in self.streams:
            if isinstance(s, Stream):
                images.append(s.image.value)
            elif isinstance(s, StreamTree):
                images.append(s.getImage(rect, mpp))


        return self.operator(images, rect, mpp, **self.kwargs)

    def getImages(self):
        """
        return a list of all the .image (which are not None)
        """
        images = []
        for s in self.streams:
            if isinstance(s, StreamTree):
                images.extend(s.getImages())
            elif isinstance(s, Stream):
                if hasattr(s, "image"):
                    im = s.image.value
                    if im is not None:
                        images.append(im)

        return images

    def getRawImages(self):
        """
        Returns a list of all the raw images used to create the final image
        """
        # TODO not sure if a list is enough, we might need to return more
        # information about how the image was built (operator, args...)
        lraw = []
        for s in self.getStreams():
            lraw.extend(s.raw)

        return lraw

    @property
    def spectrum_streams(self):
        """ Return a flat list of spectrum streams """
        return self.get_streams_by_type(SPECTRUM_STREAMS)

    @property
    def em_streams(self):
        """ Return a flat list of electron microscope streams """
        return self.get_streams_by_type(EM_STREAMS)

    def get_streams_by_name(self, name):
        """ Return a list of streams with have names that match `name` """

        leaves = set()
        for s in self.streams:
            if isinstance(s, Stream) and s.name.value == name:
                leaves.add(s)
            elif isinstance(s, StreamTree):
                leaves |= s.get_streams_by_name(name)

        return list(leaves)

    def get_streams_by_type(self, stream_types):
        """ Return a flat list of streams of `stream_type` within the StreamTree
        """
        streams = []

        for s in self.streams:
            if isinstance(s, StreamTree):
                streams.extend(s.get_streams_by_type(stream_types))
            elif isinstance(s, stream_types):
                streams.append(s)

        return streams


