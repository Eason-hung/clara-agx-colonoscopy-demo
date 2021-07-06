/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include<time.h> 

#include "deepstream_common.h"
#include "gstnvdsmeta.h"
#ifndef PLATFORM_TEGRA
#include "gst-nvmessage.h"
#endif

#include "nvbufsurface.h"
#include "gstnvdsinfer.h"

//set SAVE_VIDEO, TRUE:save the video, FALSE:EGL output
#define SAVE_VIDEO TRUE

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 320
#define MUXER_OUTPUT_HEIGHT 320

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

long time_pre=0;
int fps=0;
float avg_fps=0;
int frame_counter=0;

static GstPadProbeReturn
latency_measurement_buf_prob(GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsMetaList * l_frame = NULL;
  NvDsUserMeta *user_meta = NULL;
  NvDsInferSegmentationMeta * SegmentationMeta = NULL;
  NvDsMetaList * l_user_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  struct timeval time_now;
	gettimeofday(&time_now, NULL);
	time_t msecs_time = (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000);
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
    l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
      if (frame_meta->frame_num%10 == 0){
	fps = 10000/(msecs_time-time_pre);
        g_print("\n************frame = %d, FPS = %d**************\n",frame_meta->frame_num, fps);
	avg_fps += fps;        
	time_pre = msecs_time;
	frame_counter++;
      }
  }
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
#ifndef PLATFORM_TEGRA
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
      break;
    }
#endif
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *tee1 = NULL, *sink = NULL, *pgie = NULL, *nvsegvisual = NULL,
      *queue1, *queue2, *nvvidconv1 = NULL, *nvvidconv2 = NULL, *nvvidconv3 = NULL, *nvvidconv4, *videomixer = NULL,
      *nvosd = NULL, *tiler = NULL, *x264enc = NULL, *qtmux = NULL, *tee2 = NULL, *videoconvert = NULL, *filter1 = NULL, *filter2 = NULL ;
  GstCaps *caps1 = NULL, *caps2 = NULL;
  GstElement *transform = NULL;
  GstPad *mixer_sink_pad = NULL;
  GstPad *latency_sink_pad = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  guint i, num_sources;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 2) {
    g_printerr ("Usage: %s <uri1> \n", argv[0]);
    return -1;
  }
  num_sources = argc - 1;

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest3-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };
    GstElement *source_bin = create_source_bin (i, argv[i + 1]);

    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);
  }

  /* Use nvinfer to infer on batched frame. */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  /* Add queue elements between every two elements */
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");
  nvvidconv2 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");
  nvvidconv3 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter3");
  nvvidconv4 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter4");

  /* Create OSD to draw on the converted RGBA buffer */
  nvsegvisual = gst_element_factory_make ("nvsegvisual", "nvsegvisual");
  
  tee1 = gst_element_factory_make ("tee", "tee1");
  videomixer = gst_element_factory_make ("videomixer", "videomixer");

  filter1 = gst_element_factory_make ("capsfilter", "filter1");
  filter2 = gst_element_factory_make ("capsfilter", "filter2");

  caps1 = gst_caps_from_string ("video/x-raw, format=RGBA");
  g_object_set (G_OBJECT (filter1), "caps", caps1, NULL);
  gst_caps_unref (caps1);
  caps2 = gst_caps_from_string ("video/x-raw, format=NV12");
  g_object_set (G_OBJECT (filter2), "caps", caps2, NULL);
  gst_caps_unref (caps2);

  if (SAVE_VIDEO){
    tee2 = gst_element_factory_make ("tee", "tee2");
    videoconvert = gst_element_factory_make ("videoconvert", "converter");
    x264enc = gst_element_factory_make ("x264enc", "h264 encoder");
    qtmux = gst_element_factory_make ("qtmux", "muxer");
    sink = gst_element_factory_make ("filesink", "filesink");
    g_object_set (G_OBJECT (sink), "location", "out.mp4", NULL);
  }

  /* Finally render the osd output */
  if(prop.integrated) {
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
  }
  if (!SAVE_VIDEO){
  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
  }

  if (!pgie || !tiler || !nvvidconv1 || !nvvidconv2|| !nvvidconv3|| !nvvidconv4|| !nvsegvisual || !tee1 || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if(!transform && prop.integrated) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Configure the nvinfer element using the nvinfer config file. */
  g_object_set (G_OBJECT (pgie),
      "config-file-path", "./pgie_unet_tlt_config.txt", NULL);

  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }

  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  g_object_set (G_OBJECT (nvsegvisual), "batch-size", 1, NULL);
  g_object_set (G_OBJECT (nvsegvisual), "width", MUXER_OUTPUT_WIDTH, NULL);
  g_object_set (G_OBJECT (nvsegvisual), "height", MUXER_OUTPUT_HEIGHT, NULL);

  g_object_set (G_OBJECT (sink), "qos", 0, NULL);

  
  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  if(prop.integrated) {
    gst_bin_add_many (GST_BIN (pipeline), tee1, queue1, pgie, queue2, tiler,
        nvvidconv1, nvsegvisual, nvvidconv2, videomixer, nvvidconv3, transform, sink, nvvidconv4, tee2, videoconvert, x264enc, qtmux, filter1, filter2, NULL);
    gst_element_link(streammux, tee1);
    if (!gst_element_link_many (tee1, queue2, nvvidconv4, videomixer, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }

    if (!SAVE_VIDEO){
      if (!gst_element_link_many (tee1, queue1, pgie, tiler,
            nvvidconv1, nvsegvisual, nvvidconv2, videomixer, nvvidconv3, transform, sink, NULL)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
      }
    }

    else{
      if (!gst_element_link_many (tee1, queue1, pgie, tiler, nvvidconv1, nvsegvisual, nvvidconv2, videomixer, nvvidconv3, filter1, videoconvert, filter2, x264enc, qtmux, sink, NULL)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
      }
    }
  }
  else {
    gst_bin_add_many (GST_BIN (pipeline), tee1, queue1, pgie, queue2, tiler,
        nvvidconv1, nvsegvisual, nvvidconv2, videomixer, nvvidconv3, sink, nvvidconv4, tee2, videoconvert, x264enc, qtmux, filter1, filter2, NULL);

    gst_element_link(streammux, tee1);
    if (!gst_element_link_many (tee1, queue2, nvvidconv4, videomixer, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
    }

    if (!SAVE_VIDEO){
      if (!gst_element_link_many (tee1, queue1, pgie, tiler,
            nvvidconv1, nvsegvisual, nvvidconv2, videomixer, nvvidconv3, sink, NULL)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
      }
    }

    else{
      if (!gst_element_link_many (tee1, queue1, pgie, tiler, nvvidconv1, nvsegvisual, nvvidconv2, videomixer, nvvidconv3, filter1, videoconvert, filter2, x264enc, qtmux, sink, NULL)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
      }
    }
  }

  //set transparency of the mask when rendering
  mixer_sink_pad = gst_element_get_static_pad (videomixer, "sink_1");
  g_object_set (mixer_sink_pad, "alpha", 0.5, NULL);
  g_object_set (videomixer, "background", 1, NULL);


  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  latency_sink_pad =  gst_element_get_static_pad (pgie, "src");
  if (!latency_sink_pad)
    g_print ("Unable to get src pad\n");
  else {
    gst_pad_add_probe (latency_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        latency_measurement_buf_prob, NULL, NULL);
 }
  gst_object_unref (latency_sink_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing:");
  for (i = 0; i < num_sources; i++) {
    g_print (" %s,", argv[i + 1]);
  }
  g_print ("\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  g_print ("Average FPS = %f\n", avg_fps/frame_counter);
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
