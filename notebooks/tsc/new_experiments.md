The evaluation notebook[2_evaluation.ipynb](2_evaluation.ipynb) already carries most of the experiments needed for the 
paper, however, we need new data. For  this we use the existing structure, run some new experiments in the background, 
and swap the data. In particular, we need the following experiments and also a folder structure that follow this.

Importantly, the services must now support a queue each that accumulates frames until infinity if they are not processed
and also each frame is released immediately (or reported in our simulation) to the next service. This we can simulate 
through a counter of how many data frames are supposedly in the queue: each service starts from 0, and as soon as there
are frames in the queue, the workers collect them one by one to process them.

The standard case is that we have an arrival rate that sits around 50% utilization and poisson arrivals, so there is queuing.

### B1 (Base Case)

We need three types of applications, with at least 5 instances (i.e., different service chains) each. I want one Computer
Vision (CV) application, one that does audio processing, and another one that does Lidar or some other sort of other data.
The idea is that we have 5 instances of each application that consist of different services (e.g., detectors, processors),
and we predict the performance of this chain while using only the data from the other 4 instances (folds). This means that
every service must appear in at least 2 different instances so that we have runtime data available.
No individual benchmarks are required anymore. We need at least 300 runs per instance, we can add more later.

Ideally we create application instances where each service has a quality knobs, and we always tune it to the same value.
This means that in the CV or the audio applications, we have a resolution that determines every step, and the whole chain
is always configured in one quality. So for instance, in run #1, we configure the CV pipeline to work with 500px across
all services, and in run #2 we choose 700px. As before, we can draw these configurations through random uniform sampling,
but just as a constraint that within one run, all services have the same quality level.

### B2 (Varying levels of evidence)

Can be answered with B1 already.

### B3 (Varying depth)

I have the feeling that it is difficult to create very long service chains (~50) so I would rather go for hierarchical
architectures, as we have it in FL. I placed two links below, but the rough idea would be that we have some configuration
knobs per service, and then we make estimations about increasingly deep hierarchies (e.g., 6 levels) but importantly we also
need the ground truth so we have to run all these different hierarchies separately.

https://www.sciencedirect.com/science/article/pii/S0167739X25001736
https://arxiv.org/abs/2505.08145

### B4 (Varying arrival rates and pattern)

This is an extension of B1, because we select one application (e.g., the CV) and conduct an experiment with different
arrival rates and pattern. This means we sweep the lambda arrivals from low to high utilization, e.g., from 0.1 up to 1.0,
with 3 to 4 ticks. Additionally, in a separate experiment, we sweep also the squared coefficient of arrivals so that we either 
have completely periodic arrivals, periodic with some jitter, poisson (Ca² = 1) or batches with (Ca² = 2 or 4).

### B5 (Varying tail behavior)

Skipped for now.

### B6 (Varying percentiles)

Can be answered with B1 already.


### C1 (Varying tail behavior)

Can be answered with B1 already.


### C2 (Varying percentiles)

Can be answered with B1 already.