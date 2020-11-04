# AWS Computer Vision Jump Starter Kit
author: dylatong@amazon.com

---

This repository is a collection of content to help enable engineers and data scientists to succeed on their Computer Vision projects on AWS. The repository currently includes labs for:

1. Amazon Rekognition Custom Labels: Computer Vision AutoML for Image Classification and Object Detection
2. Amazon GroundTruth: Managing Machine Learning Annotations at Scale
3. Amazon SageMaker Built-in Algorithms: A brief follow-up to the GroundTruth Lab for building a SSD Object Detector using the SageMaker built-in Algorithm.
4. Amazon SageMaker and GluonCV for Object Detection: Training and Deploying YOLOv3 on GluonCV and SageMaker.
5. Amazon SageMaker and GluonCV for Pose Estimation: Training and Deploying an Inference Pipeline for 2D Human Pose predictions.

Lab guides are shared [here](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/tree/master/lab-guides).

2020 Overview Presentation is shared [here](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/presentations/aws-cv-partner-enablement-2020.pdf)

Related Content:
* 2019 Image Classification and Object Detection Partner [Webinar](https://www.youtube.com/watch?v=t7smVKL6d4Q)
* Image Similarity using PyTorch AWS ML [Blog](https://aws.amazon.com/blogs/machine-learning/a-personalized-shop-by-style-experience-using-pytorch-on-amazon-sagemaker-and-amazon-neptune/) (references to repository)
* Jetson Nano Smart Cam [Repository](https://github.com/dylan-tong-aws/jetson-nano-smart-cam)

### Sample Workshop Packages
Examples of modules that you can use together to tailor a workshop for your audience.

**I. Object Detection Workshop Package (est. 4 hours)**

Below is content you can package up into a Object Detection workshop for SageMaker. You can put together a 4-5 hour agenda with this content.

1. [AWS CV Introduction Presentation](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/presentations/aws-cv-partner-enablement-2020.pdf)
2. [Lab1: Ground Truth](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/lab-guides/Lab1-GroundTruth/Lab1-%20Ground%20Truth.pdf):
    - Learn to create and manage a quality data set at
    scale using SageMaker GroundTruth.
    - Manage annotation workforces: private,
    public (Mechanical Turk), and 3rd party
    vendors.
    - Create a labeling job (for Object Detection)
 3. [Lab2: SageMaker Algorithms- Object Detection](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/lab-guides/Lab2-SM-ObjectDetection/Lab2-SageMaker-Algorithms-ObjectDetection.pdf):
    - Learn to build a custom object detection (Single-shot
    Detection) from the training data you created in Lab1
    without having to write code.
    - Learn about hyper-parameter tuning automation.

4. [Lab3: Bring Your Own Script- Object Detection](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/lab-guides/Lab3-GluonCV-YOLOv3/Lab3-BYOS%20YOLOv3%20Object%20Detector%20on%20GluonCV.pdf):
    - Learn how to bring your own script from a deep
    learning framework.
    - In the lab we’ll bring a GluonCV script to train an
    object detection model (YOLOv3 on mobileNet).
    - Learn how to programmatically launch a
    hyperparameter tuning job, SageMaker local training
    as well as perform incremental training.
    - Learn how to deploy a real-time endpoint for
    inference.


**II. Complete AWS Computer Vision Workshop (est. 16 hours)**

1. [AWS CV Introduction Presentation](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/presentations/aws-cv-partner-enablement-2020.pdf)
2. [Rekognition Custom Labels Lab](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/lab-guides/Lab-Amazon%20Rekognition%20Custom%20Labels/Lab-Rekognition%20Custom%20Labels.pdf)
    - Provides a first hands-on experience with Rekognition Custom Labels through a logo detection use case.
    - Learn the iterative process and get a sense of how to extend the learnings to deliver a production-ready deployment.
3. [Object Detection Series]:
    - Learn the end-to-end experimentation process for CV projects on Amazon SageMaker
    - Learn the different ways to approach CV problems on Amazon SageMaker and acquire an understanding of the pros and cons.
    
    - [Lab1: Ground Truth](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/lab-guides/Lab1-GroundTruth/Lab1-%20Ground%20Truth.pdf):
        - Learn to create and manage a quality data set at
    scale using SageMaker GroundTruth.
        - Manage annotation workforces: private,
    public (Mechanical Turk), and 3rd party
    vendors.
        - Create a labeling job (for Object Detection)
    - [Lab2: SageMaker Algorithms- Object Detection](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/lab-guides/Lab2-SM-ObjectDetection/Lab2-SageMaker-Algorithms-ObjectDetection.pdf):
        - Learn to build a custom object detection (Single-shot
    Detection) from the training data you created in Lab1
    without having to write code.
        - Learn about hyper-parameter tuning automation.

    - [Lab3: Bring Your Own Script- Object Detection](https://github.com/dylan-tong-aws/aws-cv-jumpstarter/blob/master/lab-guides/Lab3-GluonCV-YOLOv3/Lab3-BYOS%20YOLOv3%20Object%20Detector%20on%20GluonCV.pdf):
        - Learn how to bring your own script from a deep
    learning framework.
        - In the lab we’ll bring a GluonCV script to train an
    object detection model (YOLOv3 on mobileNet).
        - Learn how to programmatically launch a
    hyperparameter tuning job, SageMaker local training
    as well as perform incremental training.
        - Learn how to deploy a real-time endpoint for
    inference.
    - [Lab4: BYOS and Inference Pipelines- Pose Estimation](https://raw.githubusercontent.com/dylan-tong-aws/aws-cv-jumpstarter/master/notebooks/lab4-gluoncv-pose-estimation-pipeline-on-sagemaker.ipynb):
        - Learn how to implement an inference pipeline of multiple Computer Vision models.
        - Learn how to build and deploy a pose estimation model.
4. [AWS CV@Edge online serices](http://cv-edge-aws.com/)
    - This is a online series with hands-on labs to teach you about CV@edge and associated tools available to you on AWS.
   
