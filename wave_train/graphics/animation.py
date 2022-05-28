import re
import os
import shutil
import logging
import subprocess
from wave_train.graphics.helper import movie
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Animation:
    """
    Class for handling creation of animated output and eventually saving the
    animation as a .mp4 file. The motivation to develop this class has been to
    solve the underlying problem of the FuncAnimation class, namely that creation
    of animation output and visualization are executed in order and not in parallel.

    1) If image_file is not None, last snaphot is saved, but previous snapshots are removed.
    2) If image_file is not None and snapshot is True, all snapshots are saved. 
       If image_file == None and snapshot == True, a default snapshot name will be assigned
    3) If image_file is not None, movie_file is not None, and snapshot is True,
       a movie is created from the snapshot and snapshots are not removed
    """
    __slots__ = [
        "dynamics",     # dynamics object used to update plot
        "movie_file",   # output name of movie 
        "movie_suffix", # file suffix of output movie
        "image_file",   # name of the image file for output of last figure state
        "image_suffix", # suffix of the image file
        "snapshots",    # avoid cleanup of files created 
        "save",         # variable holding the logical state if output movie is generated
        "fig",          # figure object holding the plot
        "update",       # update function called in FuncAnimation loop
        "init",         # init function called in FuncAnimation setup
        "frames",       # frames in the animation
        "pause",        # pause between frames
        "frame_rate",   # number of frames per second
        "frame_count"   # counter of frames/snaphots in animation
    ]

    movie_formats = ['mp4']
    image_formats = ['png']

    def __init__(self, dynamics, movie_file, image_file, snapshots, figure, update, 
                            init=None, frames=None, blit=False, pause=200, frame_rate=10):
        """
        Parameters
        ------------
        dynamics: Instance of Dyanmics class
            The instance holding the dynamics object, which is provided
            via the fargs option to the update function
        movieFile: str
            Name of the movie file saving the animation output
        imageFile: str
            Name of the image file snapshotting the last frame
        figure: Figure object
            Figure instance for animated output
        update: Callable
            Update function for new frame
        init: Callable
            Initializer function for defining plot parameters
        frames: Iterable
            Number of frames to display
        blit: bool 
            Use of blitting
        pause: uint
            Pause between each frame in visualization
        frame_rate:  uint
            Number of frames in animation output
        """
        self.dynamics       = dynamics
        self.movie_file     = movie_file
        self.image_file     = image_file
        self.snapshots      = snapshots
        self.fig            = figure
        self.update         = update
        self.init           = init
        self.frames         = frames
        self.pause          = pause
        self.frame_rate     = frame_rate
        self.frame_count    = 0
        self.save           = True

        if not (isinstance(movie_file, str) or movie_file is None):
            raise TypeError("Movie file argument must be an instance of 'str' or 'NoneType'")
        if not (isinstance(image_file, str) or image_file is None):
            raise TypeError("Image file argument must be an instance of 'str' or 'NoneType'")

        if self.image_file is None:
            if snapshots and self.movie_file is not None:
                self.image_file = "snaphot.png"
                logging.warning("Name for snapshots was not provided. Defaulting to 'snapshot.png'")
            elif self.movie_file is not None:
                # indicate that all snaphots must be removed
                self.image_file = "snapshot_removable.png"
            else:
                self.image_file = ""

        if self.movie_file is None:
            self.movie_file = ""

        self.image_file, self.image_suffix = os.path.splitext(self.image_file)
        self.movie_file, self.movie_suffix = os.path.splitext(self.movie_file)

    def start(self):
        try:
            animation = FuncAnimation(self.fig, self.update, init_func=lambda: None, frames=self.frames,
                                      fargs=(self.fig, self.dynamics, self, self.save),
                                      repeat=False)
            plt.show()
        except Exception as AnimationError:
            print(AnimationError)
        finally:
            movie_files = self._get_temp_files()

            if len(movie_files) == 0:
                return

            if self.movie_file != "":
                self._export_as_mp4(movie_files)

            if not self.snapshots and self.image_file != "": 
                # if the snapshot is removable, remove all snapshots
                if not self.image_file == "snapshot_removable":
                    movie_files.pop(-1)

                self._cleanup(movie_files)

            if self.image_file == "" and self.snapshots == False:
                self._cleanup(movie_files)

    def _export_as_mp4(self, movie_files):
        print("""
------------------------
Saving animation to file
------------------------

Filename : {}
Frames   : {} per second
        """.format(self.movie_file, str(self.frame_rate)))

        subprocess.call(["ffmpeg", "-loglevel", "quiet", "-y", "-r", str(self.frame_rate - 9), "-i",
                         f"{self.image_file}_%06d{self.image_suffix}", f"output{self.movie_suffix}"])

        subprocess.call(["ffmpeg", "-loglevel", "quiet", "-y", "-i", f"output{self.movie_suffix}", "-c:v", "libx264", "-preset",
                         "slow", "-crf", "22", "-pix_fmt", "yuv420p", "-c:a",
                         "libvo_aacenc", "-b:a", "128k", f"{self.movie_file}{self.movie_suffix}"])

        os.remove(f"output{self.movie_suffix}")

    def _get_temp_files(self):
        tmp_files = re.compile("%s_\d{6}%s" % (self.image_file, self.image_suffix))
        movie_files = []

        for root, dir, files in os.walk("."):
            for tmp in files:
                if re.match(tmp_files, tmp):
                    movie_files.append(tmp)

        return sorted(movie_files)

    def _cleanup(self, movie_files): 
        """
        Delete all temporary files that were created during the
        animation run. Ideally, the movie_files argument should 
        be provided as the return of the _get_temp_files function.
        """
        for movie_file in movie_files:
            os.remove(movie_file)

    def save_as_image(self):
        if self.image_file != "":
            file_name = f"{self.image_file}_{str(self.frame_count).zfill(6)}{self.image_suffix}"
            self.fig.savefig(file_name)
            self.frame_count += 1