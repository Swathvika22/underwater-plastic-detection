Title: Pelagic Vision: Underwater Plastic Detection & Volume Estimation

Description: An AI-powered solution to detect underwater debris and estimate its volume using YOLOv8 instance segmentation.

Tech Stack: FastAPI, Ultralytics YOLOv8, Docker, HTML/CSS.

2. Deploy to Hugging Face Spaces
Now that GitHub has the code, Hugging Face will act as your "24/7 server."

Log in to Hugging Face and click your profile icon (top right) -> New Space.

Configure the Space:

Space name: underwater-plastic-detection

Select the SDK: Choose Docker.

Visibility: Public.

Connect to GitHub:

Once the Space is created, go to the Settings tab of your new Hugging Face Space.

Scroll down to the GitHub integration section.

Click Connect to GitHub and select your underwater-plastic-detection repository.

Wait for the Build:

Hugging Face will instantly see your Dockerfile. It will automatically pull the code from GitHub and start the build process.

Check the "Build logs" tab. If everything is configured correctly, it will show a "Build successful" message within a few minutes.
