import torch
from torch_geometric.loader import DataLoader
import numpy as np
import copy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import random
from modules.utils.utils import get_config
from modules.utils.sampling import positive_sampling, construct_negative_graph
from modules.data_pipeline.RulesDataset import RulesDataset
from modules.data_pipeline.GraphDataset import GraphDataset
from modules.LinkPredictor import LinkPredictor


# Define the loss function
def compute_loss(out_pos, out_neg, gt_pos, gt_neg):
    """

    :param out_pos: model output for the graph with positive sampling
    :param out_neg: model output for the graph with negative sampling
    :param gt_pos: ground-truth for positive sampling
    :param gt_neg: ground-truth for negative sampling
    :return: loss value
    """
    pos = torch.gather(out_pos, 1, gt_pos)
    neg = torch.gather(out_neg, 1, gt_neg)
    loss_bce = - torch.log(pos + 1e-15).sum() - torch.log(1 - neg + 1e-15).sum()
    loss_bce = loss_bce / (len(pos) + len(neg))
    return loss_bce


if __name__ == '__main__':
    # read the config file with training parameters
    config = get_config("./cfg/training.yaml")

    # set seeds
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # get the rule-dataset
    rule_dataset = RulesDataset(config)
    # split the data
    rule_train, rule_val, rule_test = rule_dataset.split_dataset(rule_dataset.dataset,
                                                                 config['training']['split']['train'],
                                                                 config['training']['split']['val'])
    # create torch Dataset
    graph_dict_train = rule_dataset.create_graph_dataset(rule_train)
    graph_dict_val = rule_dataset.create_graph_dataset(rule_val)
    graph_dict_test = rule_dataset.create_graph_dataset(rule_test)
    graph_generator = GraphDataset(rule_train, graph_dict_train)
    graph_generator_val = GraphDataset(rule_val, graph_dict_val)
    graph_generator_test = GraphDataset(rule_test, graph_dict_test)

    # crate torch DataLoader
    train_loader = DataLoader(graph_generator, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(graph_generator_val, batch_size=512, shuffle=False)

    # Define the LinkPredictor model
    model = LinkPredictor()
    num_params_per_layer = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = torch.prod(torch.tensor(param.size())).item()
            num_params_per_layer[name] = num_params
    print(model)
    print(f"\nTotal number of parameters: {np.sum(np.array(list(num_params_per_layer.values())))}")

    # Define the optimizer and parameters
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate_schedule']["start_value"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=config['training']['learning_rate_schedule']['factor'],
                                  patience=config['training']['learning_rate_schedule']['patience'],
                                  verbose=True)
    best_val_loss = 1e15
    num_epochs = config['training']['num_epochs']

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        # Create a tqdm progress bar for the training data
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                            unit="batch")

        for batch_idx, inputs in progress_bar:
            # Zero the gradients
            optimizer.zero_grad()

            g_input, g_gt = positive_sampling(inputs, inputs.batch, rule_dataset.rule_encoding)
            negative_graph = construct_negative_graph(inputs, inputs.batch, rule_dataset.rule_encoding)
            out = model(inputs.x.type(torch.FloatTensor), g_input.edge_index, g_gt.edge_index, g_input.edge_attr)
            out_negative = model(inputs.x.type(torch.FloatTensor), g_input.edge_index, negative_graph.edge_index,
                            g_input.edge_attr)
            loss = compute_loss(out, out_negative, g_gt.edge_attr, negative_graph.edge_attr)
            loss.backward()

            # Update the model's parameters
            optimizer.step()
            running_loss += loss.item()

            # Update the tqdm progress bar description with the current loss
            progress_bar.set_postfix({"Loss": loss.item()})

        # Close the tqdm progress bar for this epoch
        progress_bar.close()
        # Print the average training loss for this epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {running_loss / len(train_loader)}")

        # Set the model to evaluation mode
        model.eval()
        val_loss = 0.0
        num_val_batches = len(val_loader)

        with torch.no_grad():
            for val_batch_idx, val_inputs in enumerate(val_loader):
                g_input, g_gt = positive_sampling(val_inputs, val_inputs.batch, rule_dataset.rule_encoding)
                negative_graph = construct_negative_graph(val_inputs, val_inputs.batch, rule_dataset.rule_encoding)
                out = model(val_inputs.x.type(torch.FloatTensor), g_input.edge_index, g_gt.edge_index,
                            g_input.edge_attr)
                out_negative = model(val_inputs.x.type(torch.FloatTensor), g_input.edge_index, negative_graph.edge_index,
                                g_input.edge_attr)
                loss = compute_loss(out, out_negative, g_gt.edge_attr, negative_graph.edge_attr)
                val_loss += loss.item()

        # Calculate and print the average validation loss for this epoch
        average_val_loss = val_loss / num_val_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Validation Loss: {average_val_loss}")

        if average_val_loss < best_val_loss:
            print('Saving model..')
            best_val_loss = average_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, f"./models/checkpoint_epoch_{epoch + 1}_valloss_{average_val_loss}.ckpt")
        scheduler.step(average_val_loss)

    print("Training finished.")
