# AWS Computer Vision Jumpstarter
author: dylatong@amazon.com

This repository is a collection of content to help enable engineers and data scientists to succeed on their Computer Vision projects on AWS.

**Sample Object Detection Workshop Package (est. 4 hours)**

Below is content you can package up into a Object Detection workshop for SageMaker. You can put together a 4-5 hour agenda with this content.

1. [AWS CV Introduction Presentation](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/presentations/AWS-CV-Jumpstarter-Intro.pptx)
2. [Workshop Guide](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/presentations/AWS-CV-Jumpstarter-Workshops.pptx): Use this as a sample template for the workshop. 
3. [Lab1: Ground Truth](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/lab-guides/Lab1-GroundTruth/Lab1-%20Ground%20Truth.pdf):
    - Learn to create and manage a quality data set at
    scale using SageMaker GroundTruth.
    - Manage annotation workforces: private,
    public (Mechanical Turk), and 3rd party
    vendors.
    - Create a labeling job (for Object Detection)
 4. [Lab2: SageMaker Algorithms- Object Detection](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/lab-guides/Lab2-SM-ObjectDetection/Lab2-SageMaker-Algorithms-ObjectDetection.pdf):
    - Learn to build a custom object detection (Single-shot
    Detection) from the training data you created in Lab1
    without having to write code.
    - Learn about hyper-parameter tuning automation.

5. [Lab3: Bring Your Own Script- Object Detection](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/lab-guides/Lab3-GluonCV-YOLOv3/Lab3-BYOS%20YOLOv3%20Object%20Detector%20on%20GluonCV.pdf):
    - Learn how to bring your own script from a deep
    learning framework.
    - In the lab we’ll bring a GluonCV script to train an
    object detection model (YOLOv3 on mobileNet).
    - Learn how to programmatically launch a
    hyperparameter tuning job, SageMaker local training
    as well as perform incremental training.
    - Learn how to deploy a real-time endpoint for
    inference.



-----
Progress Journal

05/15/09: Object Detection Module Completed. 3 Labs: Ground Truth, SM Object Detection Algorithm (end-to-end), and GluonCV on SageMaker (end-to-end).

---------------
Version 0.1

Journey

SageMaker Path

1. Data infrastructure basics
2. Data set -- Ground Truth
3. Notebook
4. Exploration: TBD?
5. Algorithms
  - SageMaker Algos: SSD, Image Classifier
  - BYOS: 
     - PyTorch, OpenCV
        - Git integration
     - GluonCV (future)
6. Training
   - Distributed Training
   - HPO
7. Neo-model optimization
8. Deployment
     - Auto-scaling
     - A/B
     - Performance monitoring
9. Production architectures
     - Video streaming cloud inference
     - Edge inference- smart cameras.

     
   
   
