#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Jan 23, 2020

@author: Éric Piel
'''
import logging
from odemis import model
from odemis.driver import avantes
import os
import queue
import threading
import time
import unittest

logging.basicConfig(format="%(asctime)s  %(levelname)-7s %(module)-15s: %(message)s")
logging.getLogger().setLevel(logging.DEBUG)

# Export TEST_NOHW=1 to force using only the simulator and skipping test cases
# needing real hardware
TEST_NOHW = (os.environ.get("TEST_NOHW", 0) != 0)  # Default to Hw testing

KWARGS = {"name": "spec", "role": "spectrometer", "sn": None}


class AvantesStaticTest(unittest.TestCase):
    """
    For tests which don't need a camera ready
    """

    def test_scan(self):
        """
        Check that we can do a scan.
        """
        cameras = avantes.Spectrometer.scan()
        if not TEST_NOHW:
            self.assertGreater(len(cameras), 0)

        for name, kwargs in cameras:
            logging.debug("Opening %s", name)
            dev = avantes.Spectrometer(name, "test", **kwargs)
            logging.debug("HW %s, SW %s", dev.hwVersion, dev.swVersion)
            dev.exposureTime.value = 1
            dev.terminate()


class AvantesTest(unittest.TestCase):

    # These need to be called explicitly from the child as it's not a TestCase
    @classmethod
    def setUpClass(cls):
        cls.spectrometer = avantes.Spectrometer(**KWARGS)

    @classmethod
    def tearDownClass(cls):
        cls.spectrometer.terminate()

    def setUp(self):
        self.got_image = threading.Event()
        # Add a bit of "margin" in case the previous test started/stopped acquisition,
        # and it's still on-going. Without it, if the new test case start an
        # acquisition we could immediately receive the data being acquired.
        time.sleep(0.5)

    def tearDown(self):
        pass
#         # just in case it failed
#         self.spectrometer.data.unsubscribe(self.receive_spec_image)

    def test_simple(self):
        """
        Just ensures that the device has all the VA it should
        """
        self.assertTrue(isinstance(self.spectrometer.binning.value, tuple))
        self.assertEqual(self.spectrometer.resolution.value[1], 1)
        self.assertEqual(len(self.spectrometer.shape), 3)
        self.assertGreaterEqual(self.spectrometer.shape[0], self.spectrometer.shape[1])
        self.assertGreater(self.spectrometer.exposureTime.value, 0)

    def test_acquisition(self):
        # Three single image acquisitions at different exposure times
        for exp in (self.spectrometer.exposureTime.range[0], 0.1, 1.01):
            self.spectrometer.exposureTime.value = exp
            self.assertAlmostEqual(exp, self.spectrometer.exposureTime.value)

            begin = time.time()
            data = self.spectrometer.data.get()
            duration = time.time() - begin

            self.assertGreaterEqual(duration, exp)
            self.assertEqual(data.shape[0], 1)
            self.assertEqual(data.shape[-1::-1], self.spectrometer.resolution.value)
            wl = data.metadata[model.MD_WL_LIST]
            self.assertEqual(len(wl), data.shape[1])

    def test_live_change(self):
        """
        Now modify while acquiring
        """
        exp = 0.1
        self.spectrometer.exposureTime.value = exp
        self._data = queue.Queue()

        self.spectrometer.data.subscribe(self.receive_spec_data)
        try:
            time.sleep(1)
            d = self._data.get()
            self.assertAlmostEqual(exp, d.metadata[model.MD_EXP_TIME])

            # Change exposure time
            exp = 0.3
            self.spectrometer.exposureTime.value = exp
            logging.debug("Updated exposure time to %s", exp)

            time.sleep(0.1 * 2)  # Long enough to make sure the latest image is gone
            # Empty the queue
            while True:
                try:
                    self._data.get(block=False)
                except queue.Empty:
                    break

            d = self._data.get()
            self.assertAlmostEqual(exp, d.metadata[model.MD_EXP_TIME])
        finally:
            self.spectrometer.data.unsubscribe(self.receive_spec_data)

    def test_software_trigger(self):
        """
        Check that the synchronisation with softwareTrigger works.
        Make it typical, by waiting for the data received, and then notifying
        the software trigger again after a little while.
        """
        self.spectrometer.exposureTime.value = 0.1  # s
        exp = self.spectrometer.exposureTime.value
        duration = exp * 1.1 + 0.1

        numbert = 4
        self.ccd_left = numbert  # unsubscribe after receiving

        try:
            self.spectrometer.data.synchronizedOn(self.spectrometer.softwareTrigger)
        except IOError:
            self.skipTest("Camera doesn't support synchronisation")
        self.spectrometer.data.subscribe(self.receive_auto_unsub)

        # Wait for the image
        for i in range(numbert):
            self.got_image.clear()
            self.spectrometer.softwareTrigger.notify()
            # wait for the image to be received
            gi = self.got_image.wait(duration + 5)
            self.assertTrue(gi, "image %d not received after %g s" % (i, duration + 5))
            time.sleep(i * 1)  # wait a bit to simulate some processing

        self.assertEqual(self.ccd_left, 0)
        self.spectrometer.data.synchronizedOn(None)

        # check we can still get data normally
        d = self.spectrometer.data.get()

        time.sleep(0.1)

    def receive_spec_data(self, df, d):
        self._data.put(d)
        wl = d.metadata[model.MD_WL_LIST]
        if d.shape[0] != 1:
            logging.error("Shape is %s", d.shape)
        if d.shape[1] != len(wl):
            logging.error("Shape is %s but wl has len %d", d.shape, len(wl))
        logging.debug("Received data of shape %s with mean %s, max %s",
                      d.shape, d.mean(), d.max())

    def receive_auto_unsub(self, df, d):
        self.ccd_left -= 1
        if self.ccd_left <= 0:
            df.unsubscribe(self.receive_auto_unsub)

        self.got_image.set()
        logging.debug("Received data %d of shape %s with mean %s, max %s",
                      self.ccd_left, d.shape, d.mean(), d.max())


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
