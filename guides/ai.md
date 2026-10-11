# AI

## TOC

- Evolution of AI Development

---

### 1: Evolution of AI Development

The purpose of this section is to famaliarize you with the different methods and workflows used to create AI models.
Inside this section is a flash-summary of how these methods and workflows evolved over the years.
These methods, while primordial, are still extremely relevant and even widely used across the industry.

### 1.1 The Original 'AI Models'

First, what actually classifies something as AI?
AI is an umbrella term loosley used to classify automated systems that follow a basic workflow:
  1. Take in an input
  2. Perform some process on that input
  3. Make a decision and/or action

#### 1.1.1 Rule-Based AI

The simplest form of AI is a `Rule-Based AI`. An example of one would be any NPC from your favorite game.
The NPCs look at their environment, chooses which animaitons to play, then performs those actions.
Even though every possible Input and Output has been accounted for, *unless your favorite NPC casually glitches through walls, or freezes*,
this is still technically an AI system.

#### 1.1.2 Decision Trees

Decision Trees take Rule-Based systems to the 'next-step' by training in the models decision path, and not statically defining it.
We would do this by creating a bunch of examples. Some for our NPC could look like this:
| distance | health | enemy | action |
| --- | --- | --- | --- |
| 5 | 90 | yes | attack |
| 3 | 10 | yes | flee |
| 50 | 90 | no | patrol |





While the recent AI craze arrived fast and plentiful, AI is very-far from new. Even back in the 1980s we were creating models using Neural Networks.
Researchers already understood `Neurons`, `Weights`, `Biases`, `Back Propagation`, `Gradient Descent`, etc etc.




### 1. The BIGGEST change

The first industry-wide change that took over AI Development workflows by storm, arrived in computer hardware and expanded by new libraries for that hardware.
Before this point, AI Models would run calculations as CPU bound work.
For a large model this meant running thousands if not millions of calculations, every training session.
The introduction of parallelized GPU computing massively opened the doors for how fast AI could be trained.
Now instead of running one calculation per thread in a training session, we calculate **ALL** operations at once in the same instance.

After researches began to utelize GPU bound work, NVDIA also realized the raw mathematical throughput a strong GPU could sustain.
This lead to NVDIA creating GPUs with special cores called `CUDA` cores, whose firmware makes these cores purely for vectorized mathematics,
and not for the standard graphics processing of normal GPU cores.

