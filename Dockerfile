# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application's code into the container
COPY app.py .
COPY load_to_bq.py .
# Define the command to run your script when the container starts
CMD [ "python", "./app.py" ]
