library(igraph)
library(tidyverse)
library(ggraph)
library(tidygraph)
library(ggforce)
adj_mat <- c(0, 1.2, 2, 0.7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
1.2, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
2, 2, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0.7, 2, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
1, 2, 3, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 1, 0, 2.4, 2, 2.1, 2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 2.4, 0, 1.1, 1.3, 0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 2, 1.1, 0, 1.2, 1.9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 2.1, 1.3, 1.2, 0, 1.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 2.4, 0.7, 1.9, 1.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 1.3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0, 2.3, 1.9, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.3, 2.3, 0, 3, 1.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1.9, 3, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0.8, 1.7, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.3, 0, 0, 0, 0, 0, 0, 0.3, 0, 0.1, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0, 0.9, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.9, 0.9, 2.2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.7, 3, 2.1, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0.7, 0, 1, 1.3, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 3, 1, 0, 2, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.2, 2.2, 2.1, 1.3, 2, 0 
)

the_adj_mat <- matrix(adj_mat, nc =26, nr=26)

g_init <- graph.adjacency(the_adj_mat, weighted = T)
#duplicate
df <- get.data.frame(g_init) 
#write_csv(df, "../python_code/adj_wgtd.csv")
df = df%>% group_by(from) %>%mutate( id_= cur_group_id()) %>% view()

# build the graph object
#network <- graph_from_data_frame(d=df, directed=F) 
network <- graph_from_adjacency_matrix(the_adj_mat,weighted = T, diag = F)
#graph_fr
# plot it

# working
#plot(network, layout=layout_on_grid)

graph_df <- graph_from_data_frame(df)
layout <- create_layout(graph_df, layout = 'stress')
# change the latitude here in x and y, label within group with the id
layout_edit <-
  layout %>% mutate(
    x = case_when(
      .ggraph.orig_index > 20 ~ sample(1:100, 26),
      .ggraph.orig_index <= 10 ~ sample(45:100, 26),
      .ggraph.orig_index > 10 &
        .ggraph.orig_index <= 20 ~ sample(25:120, 26)
    ),
    y = case_when(
      .ggraph.orig_index > 20 ~ sample(101:1000, 26),
      .ggraph.orig_index <= 10 ~ sample(450:1000, 26),
      .ggraph.orig_index > 10 &
        .ggraph.orig_index <= 20 ~ sample(256:1205, 26)
    ),
    id = case_when(
      .ggraph.orig_index > 20 ~ "3",
      .ggraph.orig_index > 10 &
        .ggraph.orig_index <= 20 ~ "2",
      .ggraph.orig_index <= 10 ~ "1"
      
    )
  ) %>% view()
ggraph(layout_edit) +
  geom_edge_link() +
  geom_node_point(aes(colour = id)) +
  geom_node_text(aes(label = as.character(.ggraph.orig_index))) +
  theme_classic()
head(layout)
layout$.ggraph.orig_index
