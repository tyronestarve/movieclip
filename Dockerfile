# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg build-essential portaudio19-dev libmagic1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY build/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy streamlit configuration first
COPY .streamlit ./.streamlit
# Copy the rest of the application's code
COPY . .

# Add the project root to the PYTHONPATH
ENV PYTHONPATH=/app

# Set the entrypoint for streamlit
ENTRYPOINT ["sh", "-c", "export PYTHONPATH=/app:$PYTHONPATH && streamlit run ui/pages/Main.py --server.port=$PORT"]
