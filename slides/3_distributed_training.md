# Entrainement distribué

# Principe généraux

-plusieurs process qui chacun tourne sur un GPU->Data en distribué
-NCCL et comms


Il y a 2 moyens en torch de faire des entraînements distribué, le DDP (Distributed Data Parallelism) ou le FSDP (Fully Sharded Data Parallelism).

# Distributed Data Parallelism

La méthode la plus simple et également à implémenter dans torch.\
Chaque GPU reçoit une copie du modèle et un micro-batch différent. Chaque copie va faire une forward pass sur ce micro-batch puis lors de la backward, on fait une moyenne de tous les gradients entre les différentes copie grâce a un 'all-reduce'.\

Avantage:
- Facile à implémenter et débugger

Inconvénients:
- Pas d'optimisation de communications (donc pas bien pour du multi-noeuds)
- Pas d'optimisation de la backward (aucun overlapping)
- Aucune réduction de la mémoire (nécéssite même que le modèle passe en entier sur le GPU)

# Annexe - Communications