var tracksData;

async function getRandomMusic() {
  try {
    let response = await fetch('/random_music');
    let data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching random music:', error);
    throw error;
  }
}

async function fetchData() {
  try {
    tracksData = await getRandomMusic();
    console.log(tracksData);

    new Vue({
      el: "#app",
      data() {
        return {
          audio: new Audio(),
          circleLeft: null,
          barWidth: null,
          duration: null,
          currentTime: null,
          isTimerPlaying: false,
          tracks: tracksData,
          currentTrack: null,
          currentTrackIndex: 0,
          transitionName: null,
          pollingInterval: null
        };
      },
      methods: {
        async fetchRandomMusic() {
          try {
            let response = await fetch('/random_music');
            let data = await response.json();
            if (JSON.stringify(data) !== JSON.stringify(this.tracks)) {
              this.tracks = data;
              this.currentTrackIndex = 0;
              this.currentTrack = this.tracks[this.currentTrackIndex];
              this.updateAudioSource();
            }
          } catch (error) {
            console.error('Error fetching random music:', error);
          }
        },
        updateAudioSource() {
          this.audio.pause();
          this.audio.src = this.currentTrack.source;
          this.audio.play();
        },
        play() {
          if (this.audio.paused) {
            this.audio.play();
            this.isTimerPlaying = true;
          } else {
            this.audio.pause();
            this.isTimerPlaying = false;
          }
        },
        generateTime() {
          let width = (100 / this.audio.duration) * this.audio.currentTime;
          this.barWidth = width + "%";
          this.circleLeft = width + "%";
          let durmin = Math.floor(this.audio.duration / 60);
          let dursec = Math.floor(this.audio.duration - durmin * 60);
          let curmin = Math.floor(this.audio.currentTime / 60);
          let cursec = Math.floor(this.audio.currentTime - curmin * 60);
          if (durmin < 10) {
            durmin = "0" + durmin;
          }
          if (dursec < 10) {
            dursec = "0" + dursec;
          }
          if (curmin < 10) {
            curmin = "0" + curmin;
          }
          if (cursec < 10) {
            cursec = "0" + cursec;
          }
          this.duration = durmin + ":" + dursec;
          this.currentTime = curmin + ":" + cursec;
        },
        updateBar(x) {
          let progress = this.$refs.progress;
          let maxduration = this.audio.duration;
          let position = x - progress.offsetLeft;
          let percentage = (100 * position) / progress.offsetWidth;
          if (percentage > 100) {
            percentage = 100;
          }
          if (percentage < 0) {
            percentage = 0;
          }
          this.barWidth = percentage + "%";
          this.circleLeft = percentage + "%";
          this.audio.currentTime = (maxduration * percentage) / 100;
          this.audio.play();
        },
        clickProgress(e) {
          this.isTimerPlaying = true;
          this.audio.pause();
          this.updateBar(e.pageX);
        },
        prevTrack() {
          this.transitionName = "scale-in";
          this.isShowCover = false;
          if (this.currentTrackIndex > 0) {
            this.currentTrackIndex--;
          } else {
            this.currentTrackIndex = this.tracks.length - 1;
          }
          this.currentTrack = this.tracks[this.currentTrackIndex];
          this.resetPlayer();
        },
        nextTrack() {
          this.transitionName = "scale-out";
          this.isShowCover = false;
          if (this.currentTrackIndex < this.tracks.length - 1) {
            this.currentTrackIndex++;
          } else {
            this.currentTrackIndex = 0;
          }
          this.currentTrack = this.tracks[this.currentTrackIndex];
          this.resetPlayer();
        },
        resetPlayer() {
          this.barWidth = 0;
          this.circleLeft = 0;
          this.audio.currentTime = 0;
          this.audio.src = this.currentTrack.source;
          setTimeout(() => {
            if(this.isTimerPlaying) {
              this.audio.play();
            } else {
              this.audio.pause();
            }
          }, 300);
        },
        favorite() {
          this.tracks[this.currentTrackIndex].favorited = !this.tracks[this.currentTrackIndex].favorited;
        },
        startPolling() {
          this.pollingInterval = setInterval(() => {
            this.fetchRandomMusic();
          }, 30000);
        }
      },
      created() {
        let vm = this;
        this.currentTrack = this.tracks[0];
        this.audio = new Audio();
        this.fetchRandomMusic();
        this.startPolling();
        this.audio.src = this.currentTrack.source;
        this.audio.ontimeupdate = function() {
          vm.generateTime();
          vm.updateBar();
        };
        this.audio.onloadedmetadata = function() {
          vm.generateTime();
        };
        this.audio.onended = function() {
          vm.nextTrack();
          vm.isTimerPlaying = true;
        };
      },
      beforeDestroy() {
        clearInterval(this.pollingInterval);
      }
    });
  } catch (error) {
    console.error('Error during data fetching and processing:', error);
  }
}

fetchData();
