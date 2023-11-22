# wyze-rule-recommendation

This repository showcases the second-place solution for the HuggingFace challenge hosted by Wyze Labs. 
For detailed information about the challenge, please refer to the original
[link](https://huggingface.co/spaces/competitions/wyze-rule-recommendation).

[![img.png](docs/images/img.png)](https://huggingface.co/spaces/competitions/wyze-rule-recommendation)

The challenge focuses on creating a recommendation system for smart home automation, specifically targeting 
the suggestion of rules to users. Each user possesses a diverse collection of devices spanning 16 different types, 
including Cameras, Motion Sensors, Thermostats, and more. Users can establish rules, each identified by a trigger 
device, a trigger state, an action device, and a corresponding action. 
The task at hand involves proposing new rules for users.

The dataset provided by Wyze Labs consists of:

- A training set containing actual rules defined by users. 
- Two test set splits (public and private), each containing actual 
rules with a leave-one-out design to assess the implemented system.

Each split includes a list of rules and a list of devices, along with their associations with users.

The devised solution formulates the problem as a link prediction task
and leverages a Graph Neural Network (GNN) implemented in 
PyTorch-Geometric. The model is trained using positive 
and negative sampling from the training set provided by Wyze Labs.
Significantly,  this solution builds upon the foundational approach 
outlined in [FedRule](https://arxiv.org/abs/2211.06812). However, it enhances the
overall performance by incorporating batched training and making 
distinct architectural choices. Further
insights into these improvements can be found in the [Approach](#approach) section.


The evaluation metrics for the competition is the mean reciprocal rank (MRR),
defined as

![img.png](docs/images/metrics.png)


In the competition, the presented approach attained an MRR close to 0.45, 
whereas the baseline only reached approximately 0.25.


## How to install
Install the dependencies in a virtual environment
``` bash
pip install -r requirements.tx
```


## Training
To train a new model, execute the ```train.py``` script. 
You can customize the training configuration 
by modifying the settings in the ```cfg/training.yaml``` file.

The model used in the competition is provided in ```models\best_model.ckpt```.


## Test
To test a model, execute the ```train.py``` script. 

``` bash
python test.py --mode=private --model_path="./models/best_model.ckpt"
```

## Approach

This section discuss the details of the approach used.

### Problem modelling
The problem has been framed as a link prediction task, wherein a graph is constructed for each user. The graph structure is defined as follows:

- Each device serves as a node, with each node characterized by a node feature representing the device model (e.g., Camera, Cloud) using one-hot encoding. 
- Every rule is translated in a directed edge connecting the trigger device to the action device. The edges may vary in type based on the trigger state and action state, with the dataset containing 45 trigger states and 47 actions.

### Model Architecture
For each graph, the model incorporates the following in a first component:

- Embedding layers are utilized for both trigger state and action for each edge.
- Node types are one-hot encoded.
- Edge features are aggregated per node and concatenated to node features.
- Two SageConv layers are applied to the concatenated features.

This component acts as a node embedding, 
providing an embedding vector for each node. 
At this stage, a set of links is considered for prediction. The prediction process involves:

- Concatenating the embedding vector for nodes in the edge.
- Applying a classification head with a sigmoid activation function to predict the edge probability.

At inference time, supposing to have a graph with n_nodes, the output 
will be a (n_nodes * n_nodes, num_edges_categories) tensor containing
the score for each possible edge. 
The scores are sorted and the top-50 scores are selected.

### Training approach

The entire network is trained using a positive and negative sampling
from the dataset, similarly to the base-line approach in 
[FedRule](https://arxiv.org/abs/2211.06812). However, the code has been
re-implemented in pytorch-geometric. In details:
In details:

- Positive sampling: example pairs are generated with a leave-one-out approach;
- Negative sampling: given the graph structure, an unseen edge is generated.

The loss function, as in the FedRule paper, consists in a binary cross entropy 
between positive and negative examples.





