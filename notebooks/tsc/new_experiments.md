The evaluation notebook [2_evaluation.ipynb](2_evaluation.ipynb) already carries most of the experiments needed for the 
paper, however, we need new data. For  this we use the existing structure, run some new experiments in the background, 
and swap the data. In particular, we need the following experiments and also a folder structure that follow this.

Importantly, the services must now support a queue each that accumulates frames until infinity if they are not processed
and also each frame is released immediately (or reported in our simulation) to the next service. This we can simulate 
through a counter of how many data frames are supposedly in the queue: each service starts from 0, and as soon as there
are frames in the queue, the workers collect them one by one to process them.

The standard case is that we have an arrival rate that sits around 50% utilization and poisson arrivals, so there is queuing.

50% utilization would be that processing at the slowest service takes 10ms, and we have an arrival every 20ms at the first service.

### B1 (Base Case)

We need three types of applications, with at least 5 instances (i.e., different service chains) each. I want one Computer
Vision (CV) application, one that does audio processing, and another one that does Lidar or some other sort of other data.
The idea is that we have 5 instances of each application that consist of different services (e.g., detectors, processors),
and we predict the performance of this chain while using only the data from the other 4 instances (folds). This means that
every service must appear in at least 2 different instances so that we have runtime data available.
No individual benchmarks are required anymore. We need at least 100/200/300 runs per instance, we can add more later.

Ideally chains are around 3 to 6 services long.

Chain 1: (object detection (Yolo n up to l)) --> transformation
Chain 2: (face detector (models 1,2,3,4)) --> transformation
Chain 5: object --> face; both are already known from the other

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

Could also be some hierarchical map-reduce.

#### Proposed experiment for B3

**Application: hierarchical camera analytics (map-reduce style).** It reuses the services of the CV application of B1,
so no new service is needed and the training data of every service comes from the B1 runs.
- *Leaves (map):* the detectors of B1, alternating between the object detector (Yolo n up to l) and the face detector.
- *Inner nodes and root (reduce):* the transformation service of B1, used as an aggregator: it merges the results of
  its two children into one result and passes that result on.
- *Item:* one frame, identified by a frame id. An arriving frame is sent to all leaves at the same moment. The
  end-to-end latency of a frame is the time from its arrival until the root releases its result.

**Merge rule (must be exactly this).** An aggregator *waits for the results of both children for the same frame id* and
then processes them as one item, in the order of the frame ids. One frame therefore gives one item at every service, and
every service sees the same arrival rate. This is the rule of map-reduce and of hierarchical FL (wait for all updates
of a round). The alternative, where every child result queues as its own item, doubles the arrival rate at every level
and overloads the root, so we do not use it. As everywhere else, every service is one FIFO queue with one worker.

**Hierarchies to run.** Balanced trees in which every aggregator has two children, with 1 to 6 levels. A tree with $L$
levels has $2^L - 1$ services: 1, 3, 7, 15, 31 and 63. Every tree is run separately, because we need its own ground
truth.
- Levels 1 to 4 (up to 15 services) are required.
- Levels 5 and 6 (31 and 63 services) only if the testbed has the cores for them. If it does not, replace them by a
  *narrow* tree of the same depth: at every level the aggregator joins the result of the level below with one new leaf.
  A narrow tree with $L$ levels has $2L - 1$ services (9 and 11 for 5 and 6 levels) and still has a path of $L$ services
  and $L - 1$ merges, which is what the depth experiment needs.

**Configuration per run.** As in B1: one quality level (resolution) for all services of a run, drawn uniformly at
random; the cores of every service are drawn per service. Limit the ranges so that the mean processing time of the
slowest service stays between about 10 ms and 40 ms. The reason is the run length below: a slower service forces a
lower arrival rate, and the run then needs proportionally longer for the same number of frames.

**Load.** Poisson arrivals. The arrival rate of a run is 0.5 ÷ (mean processing time of its slowest service, in
seconds), i.e. the standard case of 50% utilization at the slowest service: between 12.5 frames/s (slowest service
40 ms) and 50 frames/s (slowest service 10 ms). The mean processing times of a configuration are not known before the
run, so every run starts with a calibration phase: 200 frames at 2 frames/s, slow enough that nothing queues, from
which the mean processing time of every service is measured and the rate is set. No bursts.

**Run length: count frames, not seconds.** The ground truth is the 95th percentile of the end-to-end latency with a
confidence interval from 10 consecutive batches. A percentile is only as reliable as the number of frames beyond it:
with 5,000 measured frames, every batch has 500 frames and 25 of them lie beyond its 95th percentile, and 50 frames of
the run lie beyond the 99th percentile (needed for B6). The last campaign measured 30 s per run at 2 to 20 requests/s,
which gave only 60 to 600 frames per run; that is too few.
- Warm-up: 500 frames, not measured, so that the queues have filled to their usual level.
- Measurement: 5,000 frames. That is 100 s at 50 frames/s and 400 s (about 7 minutes) at 12.5 frames/s.
- If the time is short: 3,000 frames is the minimum (15 frames beyond the 95th percentile per batch).

**Number of runs.** 100 valid runs per hierarchy. A method counts as valid if it covers at least 95% of the runs, so
with 100 runs one miss is 1%; with the 50 runs of the last chains, three misses already decide. A run is valid if no
service was overloaded (the queue of every service is empty again at some point in the last tenth of the run).

**Expected duration.** A run takes about 4 minutes on average over the range of rates, plus the calibration (100 s)
and the deployment. With 6 minutes per run, 100 runs take 10 hours per hierarchy and 60 hours for all six. Run levels
1 to 4 first (40 hours); they already answer the question.

**What to store per run.** The same files as before, with these additions:
- per frame and service: the time the item was complete at the service (both child results there), the start and the
  end of its processing; from these we get the waiting for the other branch, the waiting in the queue and the
  processing time;
- per frame: the arrival time and the time the root released the result;
- per run: the topology as a list of edges, the configuration of every service, the arrival rate that was set, and the
  number of rows of every csv file in `summary.json`. Several files of the last copy were cut off in the middle of a
  line, and with the row count we notice that.

**Training data.** No run of a hierarchy is used to train a model for that hierarchy. The models of the detectors and
of the transformation service are trained on the B1 runs, and on the runs of the hierarchies with another number of
levels.

### B4 (Varying arrival rates and pattern)

This is an extension of B1, because we select one application (e.g., the CV) and conduct an experiment with different
arrival rates and pattern. This means we sweep the lambda arrivals from low to high utilization, e.g., from 0.1 up to 1.0,
with 3 to 4 ticks. Additionally, in a separate experiment, we sweep also the squared coefficient of arrivals so that we either 
have completely periodic arrivals (Ca² = 0), periodic with some jitter(Ca² = 0.3), poisson (Ca² = 1) or batches with (Ca² = 2 or 4).

### B5 (Varying tail behavior)

Skipped for now.

### B6 (Varying percentiles)

Can be answered with B1 already.


### C1 ()

Can be answered with B1 already.


### C2 ()

Can be answered with B1 already.