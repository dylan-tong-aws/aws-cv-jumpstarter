**Lab 1: Creating your Training Set with Ground Truth**

1.  Log into your AWS account and ensure you're in the right region
    designated for your workshop. The screenshot below indicates that
    I'm currently in us-east-1 (N. Virginia).

> ![](media/image1.png){width="2.6737576552930884in"
> height="1.2340419947506562in"}

2.  A notebook instance isn't necessary for our task, but we're going to
    start one up now for convenience, and use in proceeding labs.

> ![](media/image2.png){width="4.904166666666667in"
> height="1.0573304899387577in"}
>
> ![](media/image3.png){width="4.904255249343832in"
> height="1.526816491688539in"}
>
> ![](media/image4.png){width="4.861701662292213in"
> height="3.663416447944007in"}
>
> ![](media/image5.png){width="4.882979002624672in"
> height="1.8436373578302712in"}

[[https://github.com/dylan-tong-aws/aws-cv-jumpstarter]{.underline}](https://github.com/dylan-tong-aws/aws-cv-jumpstarter)

> ![](media/image6.png){width="4.308510498687664in"
> height="2.73874343832021in"}
>
> ![](media/image7.png){width="4.265956911636046in"
> height="0.8094378827646544in"}
>
> ![](media/image8.png){width="2.1382983377077864in"
> height="0.8303357392825896in"}
>
> ![](media/image9.png){width="2.1381944444444443in"
> height="0.7999540682414699in"}

aws s3api create-bucket \--bucket dtong-cv-jumpstarter-workshop
**\--region us-west-2 \--create-bucket-configuration
LocationConstraint=us-west-2,us-east-1**

![](media/image10.png){width="6.5in" height="0.8770833333333333in"}

> ![](media/image11.png){width="3.5106386701662293in"
> height="1.0701946631671042in"}
>
> ![](media/image12.png){width="4.292328302712161in"
> height="0.4255325896762905in"}

Replace \<\< YOUR S3 BUCKET NAME \>\> with the name of your bucket

> ![](media/image13.png){width="4.291666666666667in"
> height="0.39015201224846896in"}
>
> Run the cells
>
> ![](media/image14.png){width="6.5in" height="0.975in"}
>
> ![](media/image15.png){width="3.606383420822397in"
> height="1.3349004811898513in"}
>
> ![](media/image16.png){width="3.6808508311461066in"
> height="1.3954265091863518in"}
>
> ![](media/image17.png){width="4.808510498687664in"
> height="0.7777865266841645in"}
>
> ![](media/image18.png){width="3.7641535433070867in"
> height="2.8829779090113736in"}
>
> ...
>
> {\"source-ref\":
> \"s3://dtong-cv-jumpstarter-workshop/ground-truth-lab/images/000062a39995e348.jpg\"}
>
> {\"source-ref\":
> \"s3://dtong-cv-jumpstarter-workshop/ground-truth-lab/images/000411001ff7dd4f.jpg\"}
>
> ...
>
> ![](media/image19.png){width="3.75in" height="1.1111111111111112in"}
>
> ![](media/image20.png){width="1.7638888888888888in"
> height="3.0694444444444446in"}
>
> ![](media/image21.png){width="3.7916666666666665in"
> height="1.4444444444444444in"}
>
> ![](media/image22.png){width="6.5in" height="1.6604166666666667in"}
>
> ![](media/image23.png){width="4.138421916010499in"
> height="4.0531911636045495in"}
>
> ![](media/image24.png){width="3.138888888888889in"
> height="0.9166666666666666in"}
>
> ![](media/image25.png){width="2.888888888888889in"
> height="1.5694444444444444in"}
>
> ![](media/image26.png){width="3.341825240594926in"
> height="1.4680850831146106in"}
>
> ![](media/image27.png){width="3.6808508311461066in"
> height="0.8884809711286089in"}
>
> ![](media/image28.png){width="3.872340332458443in"
> height="0.6826235783027121in"}
>
> ![](media/image29.png){width="3.873203193350831in"
> height="1.8297867454068242in"}
>
> ![](media/image30.png){width="4.332375328083989in"
> height="2.627659667541557in"}
>
> ![](media/image31.png){width="2.4027777777777777in"
> height="1.7777777777777777in"}
>
> ![](media/image32.png){width="4.830985345581802in"
> height="4.574468503937008in"}
>
> ![](media/image33.png){width="5.032468285214348in"
> height="3.287234251968504in"}
>
> ![](media/image34.png){width="5.100075459317585in"
> height="3.308510498687664in"}
>
> ![](media/image35.png){width="6.5in" height="2.547222222222222in"}
>
> ![](media/image36.png){width="5.441026902887139in"
> height="3.1489359142607176in"}
>
> ![](media/image37.png){width="6.5in" height="1.7284722222222222in"}

Create a good and bad example. Eg) Share the two images publicly on S3.

![](media/image38.png){width="1.553332239720035in"
height="1.4574475065616799in"}

<https://dvt7olt8euncl.cloudfront.net/41473cc4-ca5a-442f-9db9-cf116e59957f/src/images/bounding-box-good-example.png>

![](media/image39.png){width="3.02127624671916in"
height="0.7071073928258967in"}

![](media/image40.png){width="1.8615332458442695in"
height="2.212765748031496in"}

<https://dvt7olt8euncl.cloudfront.net/41473cc4-ca5a-442f-9db9-cf116e59957f/src/images/bounding-box-bad-example.png>

![](media/image41.png){width="1.9583333333333333in"
height="2.513888888888889in"}

![](media/image42.png){width="4.617020997375328in"
height="1.9338834208223972in"}

![](media/image43.png){width="1.5138888888888888in"
height="1.8611111111111112in"}

![](media/image44.png){width="3.8565573053368327in"
height="3.595744750656168in"}

![](media/image45.png){width="3.25in" height="1.4027777777777777in"}

![](media/image46.png){width="6.073977471566054in" height="3.0in"}

![](media/image47.png){width="4.236961942257218in"
height="3.8404254155730535in"}

> ![](media/image48.png){width="6.5in" height="3.6534722222222222in"}
>
> ![](media/image49.png){width="5.583333333333333in"
> height="5.111111111111111in"}
>
> ![](media/image50.png){width="5.569444444444445in"
> height="6.416666666666667in"}
>
> ![](media/image51.png){width="6.5in" height="1.9201388888888888in"}
>
> \*may need to refresh the browser
>
> ![](media/image52.png){width="6.5in" height="4.063888888888889in"}
>
> ![](media/image53.png){width="6.5in" height="1.8319444444444444in"}
>
> ![](media/image54.png){width="6.5in" height="4.0784722222222225in"}
>
> ![](media/image55.png){width="5.698885608048994in"
> height="3.691489501312336in"}
>
> ![](media/image56.png){width="5.404361329833771in"
> height="3.2553193350831147in"}
>
> ![](media/image57.png){width="5.255319335083114in"
> height="1.6585706474190727in"}
>
> \*Note the delay
>
> ![](media/image58.png){width="5.404166666666667in"
> height="2.003467847769029in"}
>
> ![](media/image59.png){width="5.510638670166229in"
> height="3.547179571303587in"}
>
> ![](media/image60.png){width="5.680850831146107in"
> height="3.330216535433071in"}
>
> ![](media/image61.png){width="6.5in" height="2.3958333333333335in"}

{

\"source-ref\":
\"s3://dtong-cv-jumpstarter-workshop/ground-truth-lab/images/000062a39995e348.jpg\",

\"dtong-birds-labeling-job\": {

\"annotations\": \[

{

\"class\_id\": 0,

\"width\": 448,

\"top\": 174,

\"height\": 850,

\"left\": 142

}

\],

\"image\_size\": \[

{

\"width\": 680,

\"depth\": 3,

\"height\": 1024

}

\]

},

\"dtong-birds-labeling-job-metadata\": {

\"job-name\": \"labeling-job/dtong-birds-labeling-job\",

\"class-map\": {

\"0\": \"bird\"

},

\"human-annotated\": \"yes\",

\"objects\": \[

{

\"confidence\": 0.09

}

\],

\"creation-date\": \"2019-05-12T23:36:20.022886\",

\"type\": \"groundtruth/object-detection\"

}

}
