# Install and load required packages
install.packages(c("igraph", "torch"))
library(igraph)
library(torch)

# Directory containing the graph files
dir_path <- "C:\\Users\\moham\\OneDrive\\Desktop\\Data\\"  # Change this to your directory path

# Function to process each graph and get the embedding
process_graph <- function(file_path) {
  # Read the graph data from the file
  graph_data <- readLines(file_path)

  # Skip the first and last lines
  graph_data <- graph_data[2:(length(graph_data) - 1)]  # Removing the first and last rows

  # Extract the number of nodes and edges from the first line
  graph_info <- as.numeric(unlist(strsplit(graph_data[1], " ")))
  num_nodes <- graph_info[1]  # Number of nodes
  num_edges <- graph_info[2]  # Number of edges

  # Read edge list from remaining lines
  edges <- do.call(rbind, lapply(graph_data[2:(length(graph_data))], function(line) {
    as.numeric(strsplit(line, " ")[[1]])  # Create edges from node pairs
  }))

  # Create an undirected graph from the edge list
  g <- graph_from_edgelist(edges[, 1:2], directed = FALSE)  # Create graph with node pairs

  # Set random features between 0 and 1 for each node
  V(g)$feature <- runif(vcount(g), min = 0, max = 1)

  # Generate adjacency matrix
  adjacency_matrix <- as.matrix(as_adjacency_matrix(g, sparse = FALSE))

  # Ensure adjacency matrix is symmetric for undirected graph
  adjacency_matrix[adjacency_matrix > 0] <- exp(-adjacency_matrix[adjacency_matrix > 0])

  # Define the GraphSAGE model with Mean Aggregation
  graphsage_model <- nn_module(
    initialize = function(input_dim, hidden_dim1, hidden_dim2, output_dim) {
      self$linear1 <- nn_linear(input_dim, hidden_dim1)
      self$linear2 <- nn_linear(hidden_dim1, hidden_dim2)
      self$linear3 <- nn_linear(hidden_dim2, output_dim)
    },

    forward = function(features, adjacency_matrix) {
      # First aggregation: Mean aggregation from neighbors
      aggregated_neighbors <- torch_mm(adjacency_matrix, features)  # Aggregating neighbor features

      # Apply transformation to aggregated features
      h <- torch_relu(self$linear1(aggregated_neighbors))

      # Second layer: Apply a second transformation
      h <- torch_relu(self$linear2(h))

      # Output layer
      h <- self$linear3(h)
      return(h)
    }
  )

  # Prepare input features and adjacency tensor
  node_features <- torch_tensor(matrix(V(g)$feature, ncol = 1, nrow = vcount(g)), dtype = torch_float())
  adjacency_tensor <- torch_tensor(adjacency_matrix, dtype = torch_float())

  # Initialize model, loss function, and optimizer
  model <- graphsage_model(input_dim = 1, hidden_dim1 = 64, hidden_dim2 = 128, output_dim = 256)
  criterion <- nn_mse_loss()
  optimizer <- optim_adam(model$parameters, lr = 0.01)  # Using Adam optimizer

  # Dummy target embeddings for training (random)
  target_embeddings <- torch_randn(c(vcount(g), 256))

  # Start measuring time for the training process
  start_time <- Sys.time()

  # Training loop (100 epochs)
  for (epoch in 1:100) {
    optimizer$zero_grad()  # Zero the gradients
    output_embeddings <- model(node_features, adjacency_tensor)  # Forward pass through the model
    loss <- criterion(output_embeddings, target_embeddings)  # Compute the loss
    loss$backward()  # Backpropagate the loss
    optimizer$step()  # Update model weights

    # Print loss every 10 epochs to monitor progress
    if (epoch %% 10 == 0) {
      print(paste("Epoch", epoch, "Loss:", loss$item()))
    }
  }

  # End measuring time for the training process
  end_time <- Sys.time()

  # Calculate the time taken for training
  training_time <- end_time - start_time
  print(paste("Time taken for training the model for graph", basename(file_path), ":", training_time))

  # Extract node embeddings
  node_embeddings <- model(node_features, adjacency_tensor)$detach()

  # Compute the graph embedding by averaging node embeddings
  graph_embedding <- node_embeddings$mean(dim = 1)  # Mean across nodes (dim=1 corresponds to averaging rows)

  # Convert the graph embedding to an R-compatible format (matrix or array)
  graph_embedding_array <- as_array(graph_embedding)

  # Return the graph embedding as a vector
  return(list(graph_embedding_array, training_time))
}

# Get the list of .txt graph files in the directory
graph_files <- list.files(dir_path, pattern = "\\.txt$", full.names = TRUE)

# Initialize a list to store embeddings and their corresponding graph IDs
all_embeddings <- list()
total_time <- 0  # Initialize a variable to accumulate the total training time

# Process each file and collect the embeddings
for (graph_file in graph_files) {  # Use 'graph_file' to refer to each individual file
  print(paste("Processing file:", graph_file))

  # Get the graph embedding for the current file and its training time
  result <- process_graph(graph_file)
  graph_embedding <- result[[1]]
  training_time <- result[[2]]

  # Extract the graph ID (file name without extension)
  graph_id <- tools::file_path_sans_ext(basename(graph_file))

  # Check if the graph embedding is valid
  if (length(graph_embedding) > 0) {
    # Add the graph ID as a new column to the embeddings list
    graph_embedding_with_id <- c(graph_id, graph_embedding)
    all_embeddings[[length(all_embeddings) + 1]] <- graph_embedding_with_id
    total_time <- total_time + training_time  # Accumulate the total training time
  } else {
    print(paste("Empty embedding for file:", graph_file))
  }
}

# Combine all embeddings into a single matrix
if (length(all_embeddings) > 0) {
  embeddings_matrix <- do.call(rbind, all_embeddings)

  # Convert the embeddings matrix to a data frame
  embeddings_df <- as.data.frame(embeddings_matrix)

  # Assign column names (first column is graph ID, the rest are the embeddings)
  colnames(embeddings_df) <- c("GraphID", paste0("ATT", 1:(ncol(embeddings_df) - 1)))

  # Save the embeddings to a CSV file
  write.csv(embeddings_df, "C:\\Users\\moham\\OneDrive\\Desktop\\all_graph_embeddings.csv", row.names = FALSE)

  # Print a message confirming the save
  print("Embeddings for all graphs saved to 'all_graph_embeddings.csv'")

  print(paste("Average training time per graph: ", total_time, " seconds"))
} else {
  print("No embeddings generated, check the processing steps.")
}
