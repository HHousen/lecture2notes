import os
import json
import logging
from argparse import Namespace
from shutil import rmtree
from functools import wraps
from pathlib import Path
from .helpers import *
from timeit import default_timer as timer
from .spell_check import SpellChecker
from .summarization_approaches import (
    full_sents,
    keyword_based_ext,
    get_complete_sentences,
    generic_abstractive,
    cluster,
    generic_extractive_sumy,
    structured_joined_sum,
)

# Step extract frames imports
from .frames_extractor import extract_frames

# Step classify slides imports
from .slide_classifier import classify_frames

# Step black border removal imports
from . import imghash
from . import border_removal
from .helpers import frame_number_from_filename

# Step perspective crop imports
from . import sift_matcher
from . import corner_crop_transform

# Step cluster slides imports
from .cluster import ClusterFilesystem
from .segment_cluster import SegmentCluster

# Step slide structure analysis imports
from . import slide_structure_analysis

# Step extract figures imports
from . import figure_detection

# Step transcribe audio imports
from .transcribe import transcribe_main as transcribe

logger = logging.getLogger(__name__)


def time_this(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = timer()
        function_outputs = f(*args, **kwargs)
        end_time = timer() - start_time
        return function_outputs, end_time

    return decorated_function


class LectureSummarizer:
    def __init__(self, params, **kwargs):
        if type(params) is str or type(params) is Path:
            with open(params, "r") as json_file:
                params = json.load(json_file)

        if type(params) is dict:
            # root_process_folder
            params = Namespace(**params)

        for name in kwargs:
            setattr(params, name, kwargs[name])

        # Perform argument checks
        if (
            params.transcription_method == "deepspeech"
            or params.transcription_method == "vosk"
        ) and params.transcribe_model_dir is None:
            logger.error(
                "DeepSpeech and Vosk methods requires --transcribe_model_dir to be set to the directory containing the deepspeech/vosk models. See the documentation for details."
            )

        if (params.summarization_mods is not None) and (
            "none" in params.summarization_mods and len(params.summarization_mods) > 1
        ):  # None and another option were specified
            logger.error(
                "If 'none' is specified in --summarization_mods then no other options can be selected."
            )

        self.all_step_functions = [
            self.step_extract_frames,
            self.step_classify_slides,
            self.step_black_border_removal,
            self.step_perspective_crop,
            self.step_cluster_slides,
            self.step_slide_structure_analysis,
            self.step_extract_figures,
            self.step_transcribe_audio,
            self.step_summarize,
        ]

        if params.spell_check:
            self.spell_checker = SpellChecker()

        self.final_data = {
            "structured_summary": None,
            "lecture_summary": None,
            "transcript": None,
        }

        self.params = params
        self.transcription_method_default = "vosk"
        self.root_process_folder = self.determine_root_path()

    def determine_root_path(self):
        if self.params.process_dir == "automatic":
            self.root_process_folder = Path(os.path.dirname(self.params.video_path))
        else:
            self.root_process_folder = Path(self.params.process_dir)
        if self.params.auto_id:
            unique_id = gen_unique_id(self.params.video_path, 12)
            self.root_process_folder = self.root_process_folder / unique_id
        if self.params.custom_id:
            self.root_process_folder = self.root_process_folder / self.params.custom_id
        return self.root_process_folder

    @time_this
    def run_all(self):
        if self.params.skip_to <= 1:
            end_time = self.step_extract_frames()[1]
            logger.info("Stage 1 (Extract Frames) took %s", end_time)
        if self.params.skip_to <= 2:
            end_time = self.step_classify_slides()[1]
            logger.info("Stage 2 (Classify Slides) took %s", end_time)
        if self.params.skip_to <= 3:
            end_time = self.step_black_border_removal()[1]
            logger.info("Stage 3 (Border Removal) took %s", end_time)
        if self.params.skip_to <= 4:
            end_time = self.step_perspective_crop()[1]
            logger.info("Stage 4 (Perspective Crop) took %s", end_time)
        if self.params.skip_to <= 5:
            end_time = self.step_cluster_slides()[1]
            logger.info("Stage 5 (Cluster Slides) took %s", end_time)
        if self.params.skip_to <= 6:
            end_time = self.step_slide_structure_analysis()[1]
            logger.info("Stage 6 (SSA and OCR Slides) took %s", end_time)
        if self.params.skip_to <= 7:
            end_time = self.step_extract_figures()[1]
            logger.info("Stage 7 (Extract Figures) took %s", end_time)
        if self.params.skip_to <= 8:
            end_time = self.step_transcribe_audio()[1]
            logger.info("Stage 8 (Transcribe Audio) took %s", end_time)
        if self.params.skip_to <= 9:
            end_time = self.step_summarize()[1]
            logger.info("Stage 9 (Summarization) took %s", end_time)

        if self.params.remove:
            rmtree(self.root_process_folder)

    @time_this
    def step_extract_frames(self):
        output_path = getattr(
            self.params,
            "extract_frames_output_path",
            self.root_process_folder / "frames",
        )
        extract_frames(
            self.params.video_path,
            self.params.extract_frames_quality,
            output_path,
            self.params.extract_every_x_seconds,
        )

    @time_this
    def step_classify_slides(self):
        frames_dir = getattr(
            self.params, "frames_dir", self.root_process_folder / "frames"
        )
        frames_sorted_dir, _, _ = classify_frames(
            frames_dir, model_path=self.params.slide_classifier_model_path
        )
        self.frames_sorted_dir = frames_sorted_dir

    @time_this
    def step_black_border_removal(self):
        self.frames_sorted_dir = getattr(
            self, "frames_sorted_dir", self.root_process_folder / "frames_sorted"
        )

        slides_dir = getattr(
            self.params, "slides_dir", self.frames_sorted_dir / "slide"
        )
        slides_noborder_dir = getattr(
            self.params,
            "slides_noborder_dir",
            self.frames_sorted_dir / "slides_noborder",
        )

        # Save first 'slide' frame number
        first_frame_num_file_path = getattr(
            self.params,
            "first_frame_num_file_path",
            self.root_process_folder / "first-frame-num.txt",
        )
        first_slide_frame_filename = sorted(os.listdir(slides_dir))[0]
        self.first_slide_frame_num = frame_number_from_filename(
            first_slide_frame_filename
        )
        with open(first_frame_num_file_path, "a") as first_frame_num_file:
            first_frame_num_file.write(self.first_slide_frame_num)

        if os.path.exists(slides_dir):
            os.makedirs(slides_noborder_dir, exist_ok=True)

            if self.params.remove_duplicates:
                images_hashed = imghash.sort_by_duplicates(slides_dir)
                imghash.remove_duplicates(slides_dir, images_hashed)

            removed_borders_paths = border_removal.all_in_folder(slides_dir)
            copy_all(removed_borders_paths, slides_noborder_dir)

        self.slides_noborder_dir = slides_noborder_dir

    @time_this
    def step_perspective_crop(self):
        self.frames_sorted_dir = getattr(
            self, "frames_sorted_dir", self.root_process_folder / "frames_sorted"
        )
        self.slides_noborder_dir = getattr(
            self, "slides_noborder_dir", self.frames_sorted_dir / "slides_noborder"
        )

        presenter_slide_dir = getattr(
            self.params,
            "presenter_slide_dir",
            self.frames_sorted_dir / "presenter_slide",
        )
        imgs_to_cluster_dir = getattr(
            self.params,
            "imgs_to_cluster_dir",
            self.frames_sorted_dir / "imgs_to_cluster",
        )

        if os.path.exists(presenter_slide_dir):
            if self.params.remove_duplicates:
                logger.info(
                    "Stage 4 (Duplicate Removal & Perspective Crop): Remove 'presenter_slide' duplicates"
                )
                imghash_start_time = timer()

                images_hashed = imghash.sort_by_duplicates(presenter_slide_dir)
                imghash.remove_duplicates(presenter_slide_dir, images_hashed)

                imghash_end_time = timer() - imghash_start_time
                logger.info(
                    "Stage 4 (Duplicate Removal & Perspective Crop): Remove 'presenter_slide' duplicates took %s",
                    imghash_end_time,
                )

            logger.info("Stage 4 (Duplicate Removal & Perspective Crop): SIFT Matching")
            siftmatch_start_time = timer()

            (
                non_unique_presenter_slides,
                transformed_image_paths,
            ) = sift_matcher.match_features(
                self.slides_noborder_dir, presenter_slide_dir
            )
            siftmatch_end_time = timer() - siftmatch_start_time
            logger.info(
                "Stage 4 (Duplicate Removal & Perspective Crop): SIFT Matching took %s",
                siftmatch_end_time,
            )

            # Remove all 'presenter_slide' images that are duplicates of 'slide' images
            # and all 'slide' images that are better represented by a 'presenter_slide' image
            for x in non_unique_presenter_slides:
                try:
                    os.remove(x)
                except OSError:
                    pass

            # If there are transformed images then the camera motion was steady and we
            # do not have to run `corner_crop_transform`. If camera motion was detected
            # then the `transformed_image_paths` list will be empty and `presenter_slide_dir`
            # will contain potentially unique 'presenter_slide' images that do not appear
            # in any images of the 'slide' class.
            if transformed_image_paths:
                copy_all(transformed_image_paths, imgs_to_cluster_dir)
            else:
                logger.info(
                    "Stage 4 (Duplicate Removal & Perspective Crop): Corner Crop Transform"
                )
                cornercrop_start_time = timer()

                cropped_imgs_paths = corner_crop_transform.all_in_folder(
                    presenter_slide_dir, remove_original=False
                )
                copy_all(cropped_imgs_paths, imgs_to_cluster_dir)

                cornercrop_end_time = timer() - cornercrop_start_time
                logger.info(
                    "Stage 4 (Duplicate Removal & Perspective Crop): Corner Crop Transform took %s",
                    cornercrop_end_time,
                )

    @time_this
    def step_cluster_slides(self):
        self.frames_sorted_dir = getattr(
            self, "frames_sorted_dir", self.root_process_folder / "frames_sorted"
        )
        self.imgs_to_cluster_dir = getattr(
            self, "imgs_to_cluster", self.frames_sorted_dir / "imgs_to_cluster"
        )
        self.slides_noborder_dir = getattr(
            self, "slides_noborder_dir", self.frames_sorted_dir / "slides_noborder"
        )

        copy_all(self.slides_noborder_dir, self.imgs_to_cluster_dir)

        if self.params.remove_duplicates:
            images_hashed = imghash.sort_by_duplicates(self.imgs_to_cluster_dir)
            imghash.remove_duplicates(self.imgs_to_cluster_dir, images_hashed)

        if self.params.cluster_method == "normal":
            cluster_filesystem = ClusterFilesystem(
                self.imgs_to_cluster_dir,
                algorithm_name="affinity_propagation",
                preference=-8,
                damping=0.72,
                max_iter=1000,
                model_path=self.params.slide_classifier_model_path,
            )
            cluster_filesystem.extract_and_add_features()
            if self.params.tensorboard:
                cluster_filesystem.visualize(self.params.tensorboard)
            cluster_dir, best_samples_dir = cluster_filesystem.transfer_to_filesystem()
        elif self.params.cluster_method == "segment":
            segment_cluster = SegmentCluster(
                self.imgs_to_cluster_dir,
                model_path=self.params.slide_classifier_model_path,
            )
            segment_cluster.extract_and_add_features()
            cluster_dir, best_samples_dir = segment_cluster.transfer_to_filesystem()
        else:
            print("Invalid `cluster_method` option")

        self.cluster_dir = cluster_dir
        self.best_samples_dir = best_samples_dir

    @time_this
    def step_slide_structure_analysis(self):
        self.frames_sorted_dir = getattr(
            self, "frames_sorted_dir", self.root_process_folder / "frames_sorted"
        )
        self.cluster_dir = getattr(
            self, "cluster_dir", self.frames_sorted_dir / "slide_clusters"
        )
        self.best_samples_dir = getattr(
            self, "best_samples_dir", self.cluster_dir / "best_samples"
        )

        ocr_raw_output_file = getattr(
            self.params,
            "ocr_raw_output_file",
            self.root_process_folder / "slide-ocr.txt",
        )
        ocr_json_output_file = getattr(
            self.params,
            "ocr_json_output_file",
            self.root_process_folder / "slide-ssa.json",
        )

        ocr_raw_text, ocr_json_data = slide_structure_analysis.all_in_folder(
            self.best_samples_dir
        )
        if "ocr" in self.params.spell_check:
            ocr_raw_text = self.spell_checker.check_all(ocr_raw_text)
        slide_structure_analysis.write_to_file(
            ocr_raw_text, ocr_json_data, ocr_raw_output_file, ocr_json_output_file
        )

        self.ocr_raw_output_file = ocr_raw_output_file
        self.ocr_json_output_file = ocr_json_output_file

    @time_this
    def step_extract_figures(self):
        self.frames_sorted_dir = getattr(
            self, "frames_sorted_dir", self.root_process_folder / "frames_sorted"
        )
        self.cluster_dir = getattr(
            self, "cluster_dir", self.frames_sorted_dir / "slide_clusters"
        )
        self.best_samples_dir = getattr(
            self, "best_samples_dir", self.cluster_dir / "best_samples"
        )
        self.ocr_json_output_file = getattr(
            self, "ocr_json_output_file", self.root_process_folder / "slide-ssa.json"
        )

        figures_dir = getattr(
            self.params, "figures_dir", self.cluster_dir / "best_samples_figures"
        )
        os.makedirs(figures_dir, exist_ok=True)

        figure_paths = figure_detection.all_in_folder(
            self.best_samples_dir, east=self.params.east_path
        )
        copy_all(figure_paths, figures_dir, move=True)

        if os.path.isfile(self.ocr_json_output_file):
            with open(self.ocr_json_output_file, "r") as ssa_json_file:
                ssa = json.load(ssa_json_file)
                ssa = figure_detection.add_figures_to_ssa(ssa, figures_dir)
            with open(self.ocr_json_output_file, "w") as ssa_json_file:
                json.dump(ssa, ssa_json_file)

        self.figures_dir = figures_dir

    @time_this
    def step_transcribe_audio(self):
        extract_from_video = self.params.video_path
        audio_path = getattr(
            self.params, "audio_path", self.root_process_folder / "audio.wav"
        )
        transcript_output_file = getattr(
            self.params,
            "transcript_output_file",
            self.root_process_folder / "audio.txt",
        )
        transcript_json_output_file = getattr(
            self.params,
            "transcript_json_output_file",
            self.root_process_folder / "audio.json",
        )

        transcript_json = None

        yt_transcription_failed = False

        if self.params.transcription_method == "youtube":
            yt_output_file = self.root_process_folder / "audio.vtt"
            try:
                transcript_path = transcribe.get_youtube_transcript(
                    self.params.video_id, yt_output_file
                )
                transcript = transcribe.caption_file_to_string(transcript_path)
            except:
                yt_transcription_failed = True
                self.params.transcription_method = self.transcription_method_default
                logger.error(
                    "Error detected in grabbing transcript from YouTube. Falling back to "
                    + self.transcription_method_default
                    + " transcription."
                )

        if self.params.transcription_method != "youtube" or yt_transcription_failed:
            transcribe.extract_audio(extract_from_video, audio_path)
            try:
                if self.params.chunk == "silence":
                    chunk_dir = self.root_process_folder / "chunks"
                    transcribe.chunk_by_silence(audio_path, chunk_dir)
                    transcript, transcript_json = transcribe.process_chunks(
                        chunk_dir,
                        model_dir=self.params.transcribe_model_dir,
                        method=self.params.transcription_method,
                    )

                    if self.params.transcribe_segment_sentences:
                        transcript, transcript_json = transcribe.segment_sentences(
                            transcript, transcript_json
                        )

                elif self.params.chunk == "speech":
                    stt_model = transcribe.load_model(
                        self.params.transcription_method,
                        self.params.transcribe_model_dir,
                    )

                    # Only DeepSpeech has a `sampleRate()` method but `stt_model` could contain
                    # a DeepSpeech or Vosk model
                    try:
                        desired_sample_rate = stt_model.sampleRate()
                    except AttributeError:
                        # default sample rate to convert to is 16000
                        desired_sample_rate = 16000

                    segments, _, audio_length = transcribe.chunk_by_speech(
                        audio_path, desired_sample_rate=desired_sample_rate
                    )
                    transcript, transcript_json = transcribe.process_segments(
                        segments,
                        stt_model,
                        method=self.params.transcription_method,
                        audio_length=audio_length,
                        do_segment_sentences=self.params.transcribe_segment_sentences,
                    )

                else:  # if not chunking
                    transcript, transcript_json = transcribe.transcribe_audio(
                        audio_path,
                        method=self.params.transcription_method,
                        model=self.params.transcribe_model_dir,
                    )

                    if self.params.transcribe_segment_sentences:
                        transcript, transcript_json = transcribe.segment_sentences(
                            transcript, transcript_json
                        )

            except Exception as e:
                logger.error(
                    "Audio transcription failed. Retry by running this script with the skip_to parameter set to 6."
                )
                raise

        if "transcript" in self.params.spell_check:
            transcript = self.spell_checker.check(transcript)

        transcribe.write_to_file(
            transcript,
            transcript_output_file,
            transcript_json,
            transcript_json_output_file,
        )

        self.transcript = transcript
        self.final_data["transcript"] = transcript
        self.transcript_output_file = transcript_output_file
        self.transcript_json_output_file = transcript_json_output_file

    @time_this
    def step_summarize(self):
        self.ocr_raw_output_file = getattr(
            self, "ocr_raw_output_file", self.root_process_folder / "slide-ocr.txt"
        )
        with open(self.ocr_raw_output_file, "r") as ocr_file:
            ocr_results_flat = ocr_file.read()
        self.ocr_json_output_file = getattr(
            self, "ocr_json_output_file", self.root_process_folder / "slide-ssa.json"
        )

        self.transcript_output_file = getattr(
            self, "transcript_output_file", self.root_process_folder / "audio.txt"
        )
        if not hasattr(self, "transcript"):
            with open(self.transcript_output_file, "r") as transcript_file:
                self.transcript = transcript_file.read()
        self.transcript_json_output_file = getattr(
            self, "transcript_json_output_file", self.root_process_folder / "audio.json"
        )
        self.extract_every_x_seconds = getattr(self, "extract_every_x_seconds", 1)

        lecture_summarized_output_file = self.root_process_folder / "summarized.txt"
        lecture_summarized_structured_output_file = (
            self.root_process_folder / "summarized.json"
        )

        ocr_results_flat = ocr_results_flat.replace("\n", " ").replace(
            "\r", ""
        )  # remove line breaks

        if (
            self.params.summarization_structured != "none"
            and self.params.summarization_structured is not None
        ):
            logger.info("Stage 9 (Summarization): Structured Summarization")
            ss_start_time = timer()

            if self.params.summarization_structured == "structured_joined":
                # Get the frame number of the first 'slide'
                if hasattr(self, "first_slide_frame_num"):
                    first_slide_frame_num = self.first_slide_frame_num
                else:
                    first_frame_num_file_path = getattr(
                        self.params,
                        "first_frame_num_file_path",
                        self.root_process_folder / "first-frame-num.txt",
                    )
                    with open(first_frame_num_file_path, "r") as first_frame_num_file:
                        first_slide_frame_num = first_frame_num_file.read()

                structured_summary = structured_joined_sum(
                    self.ocr_json_output_file,
                    self.transcript_json_output_file,
                    frame_every_x=self.extract_every_x_seconds,
                    ending_char=".",
                    first_slide_frame_num=int(first_slide_frame_num),
                    to_json=lecture_summarized_structured_output_file,
                    summarization_method=self.params.structured_joined_summarization_method,
                    abs_summarizer=self.params.structured_joined_abs_summarizer,
                    ext_summarizer=self.params.structured_joined_ext_summarizer,
                    hf_inference_api=self.params.abs_hf_api,
                )
                self.final_data["structured_summary"] = structured_summary

            ss_end_time = timer() - ss_start_time
            logger.info("Stage 9 (Summarization): Structured took %s", ss_end_time)
        else:
            logger.info("Skipping structured summarization.")

        # Combination Algorithm
        logger.info("Stage 9 (Summarization): Combination Algorithm")
        if self.params.combination_algo == "only_asr":
            summarized_combined = self.transcript
        elif self.params.combination_algo == "only_slides":
            summarized_combined = ocr_results_flat
        elif self.params.combination_algo == "concat":
            summarized_combined = ocr_results_flat + self.transcript
        elif self.params.combination_algo == "full_sents":
            summarized_combined = full_sents(ocr_results_flat, self.transcript)
        elif self.params.combination_algo == "keyword_based":
            summarized_combined = keyword_based_ext(ocr_results_flat, self.transcript)
        else:  # if no combination algorithm was specified, which should never happen since argparse checks
            logger.warn("No combination algorithm selected. Defaulting to `concat`.")
            summarized_combined = ocr_results_flat + self.transcript

        # Modifications
        logger.info("Stage 9 (Summarization): Modifications")
        if (
            self.params.summarization_mods != "none"
            and self.params.summarization_mods is not None
        ):
            if "full_sents" in self.params.summarization_mods:
                summarized_mod = get_complete_sentences(
                    summarized_combined, return_string=True
                )
        else:
            summarized_mod = summarized_combined
            logger.debug("Skipping summarization_mods")

        # Extractive Summarization
        logger.info("Stage 9 (Summarization): Extractive")
        ext_start_time = timer()
        if (
            self.params.summarization_ext != "none"
            and self.params.summarization_ext is not None
        ):  # if extractive method was specified
            if self.params.summarization_ext == "cluster":
                summarized_ext = cluster(
                    summarized_mod,
                    title_generation=False,
                    cluster_summarizer="abstractive",
                    hf_inference_api=self.params.abs_hf_api,
                )
            else:  # one of the generic options was specified
                summarized_ext = generic_extractive_sumy(
                    summarized_mod, algorithm=self.params.summarization_ext
                )
        else:
            logger.debug("Skipping summarization_ext")
        ext_end_time = timer() - ext_start_time
        logger.info("Stage 9 (Summarization): Extractive took %s", ext_end_time)

        # Abstractive Summarization
        logger.info("Stage 9 (Summarization): Abstractive")
        abs_start_time = timer()
        if (
            self.params.summarization_abs != "none"
            and self.params.summarization_abs is not None
        ):  # if abstractive method was specified
            lecture_summarized = generic_abstractive(
                summarized_ext,
                self.params.summarization_abs,
                hf_inference_api=self.params.abs_hf_api_overall,
            )
        else:  # if no abstractive summarization method was specified
            lecture_summarized = summarized_ext
            logger.debug("Skipping summarization_abs")
        abs_end_time = timer() - abs_start_time
        logger.info("Stage 9 (Summarization): Abstractive took %s", abs_end_time)

        transcribe.write_to_file(lecture_summarized, lecture_summarized_output_file)

        self.final_data["lecture_summary"] = lecture_summarized
