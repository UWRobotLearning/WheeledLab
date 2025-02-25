document.addEventListener('DOMContentLoaded', function() {
  // Function to set up video selection for a specific section
  function setupVideoSection(sectionElement) {
    const videoSelector = sectionElement.querySelector('.select select');
    const videoContainer = sectionElement.querySelector('.column .has-text-centered');

    if (!videoSelector || !videoContainer) return;

    videoSelector.addEventListener('change', function() {
      // Hide all video figures in this section's left column
      videoContainer.querySelectorAll('.video-figure').forEach(figure => {
        figure.style.display = 'none';
      });

      // Show the selected video figure in the left column
      const selectedValue = this.value;
      const selectedFigure = sectionElement.querySelector(`#figure-${selectedValue}`);
      if (selectedFigure) {
        selectedFigure.style.display = 'block';
      }

      // Pause only the videos in the left column of this section
      videoContainer.querySelectorAll('video').forEach(video => {
        if (video.id !== selectedValue) {
          video.pause();
        }
      });

      // Play the selected video in the left column
      const selectedVideo = document.getElementById(selectedValue);
      if (selectedVideo) {
        selectedVideo.play();
      }
    });
  }

  // Find all sections and set up each one
  const sections = document.querySelectorAll('section.section');
  sections.forEach(setupVideoSection);
});