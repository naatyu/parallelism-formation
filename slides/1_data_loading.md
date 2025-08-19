# Chargement des données

Une des premières étapes de notre entraînement c'est le chargement des données. C'est donc également un des premiers bottlenecks possible dans notre entraînement. On est également vite confronté à un mur lorsque l'on veut manipuler des datasets massifs (plusieurs TB de données).
Avoir un chargement des données efficace et optimisé est donc une première étape éssentielle.

# 1 - Map Dataset

Les deux principaux éléments de PyTorch pour le chargement des données sont le `Dataset` (disponible sous 2 versions mais on vera cela par la suite...) et le `Dataloader`.
La classe `Dataset` est assez simple:
```py
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```
On accède aux données par un index, ce type de dataset est appelé 'Map Dataset'. Dans ce type de datasets les données sont généralement en RAM (ce qui nécéssite que le dataset passe en RAM...). Il est basique mais assez efficace pour des petits/moyens datasets.

# 2 - Dataloader

Le `Dataloader` est lui beaucoup plus intéréssant, nottament à cause des nombreux arguments incompris qu'il possède:
```py
torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=None,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    *,
    prefetch_factor=None,
    persistent_workers=False,
    in_order=True)
```
Certains valent le coup qu'on s'y penche dessus.

## 2.1 - Batch size

C'est un argument assez simple, il correspond au nombre de samples du dataset que le dataloader doit récupérer. Pourtant il peut avoir un impact assez fort dû à l'architecture actuelle des GPUs.

<img src="1_data_loading_images/warp.png" alt="gpu warp" width="300"/>

Un GPU possède des *warps*. Un warp est composé d'un certains nombre de threads (32 pour les architectures récentes) et tous les threads d'un warp sont exécutés en même temps.\
Ce qui veut dire que si on a un kernel (fonction qui tourne sur un GPU) qui à besoin que d'un seul thread, tous les autres threads (31 autres) seront en stand by et inutilisable tant que ce thread n'aura pas finit son exécution.\
C'est pour cela qu'on favorise une batch size multiple de 32 car cela correspond au nombre de threads dans un warp. On peut utiliser des multiples de 16 ou 8 mais il vaut mieux éviter d'aller au delà, cela peut réduire les performances.

## 2.2 - Workers

Un worker est un processus qui va s'occuper du chargement des données. L'avantage c'est que on peux donc avoir plusieurs workers qui travaillent en même temps pour charger la donnée.\
Si on a `num_wokers=0` alors le processus principal (celui qui s'occupe également de tout le reste dans notre entraînement) va s'occuper de charger les données. En revanche pour `num_workers=N` on va avoir $N$ différents processus qui vont s'occuper de charger les données, le processus principal reste focaliser sur le reste de notre entraînement.

<img src="1_data_loading_images/workers.png" alt="gpu warp" width="400"/>

En revanche ce n'est pas parfait:
- Comme la mémoire est partagé entre les processus, cela peut créer des accès concurentiels sur nos données et donc un potentiel bottleneck.
- Ajouter des workers revient aussi à augmenter la RAM nécéssaire ainsi que les opérations de communication (les processus utilise l'Inter Process Communication).
- Avoir trop de workers n'est également pas une bonne chose car cela entraine une forte utilisation CPU et donc un effet que l'on appelle le [*Noisy Neighbor*](https://facebookresearch.github.io/spdl/latest/optimization_guide/noisy_neighbour.html). Lorsque l'utilisation CPU est trop forte (à partir de 75% d'utilisation moyenne entre tous les coeurs), les coeurs CPU sont trop occupés à gérer la donnée et donc ont moins de temps pour lancer des kernels. Cela va donc ralentir notre entraînement même si on a beaucoup de workers qui chargent nos données.

Une bonne valeur se trouve en 2 et 6 workers de manière générale. Un benchmark rapide, quelques itérations sur votre boucle d'entraîment, peut lever le doute.

## 2.3 - Memory Pinning

Avant de continuer la lecture de cette section qui aborde des méchanismes sur la mémoire, allez lire l'annexe [RAM et Mémoire virtuelle](#annexe---ram-et-mémoire-virtuelle) si vous n'êtes pas familier avec son fonctionnement. 

# Section sur notre infra et comment elle peut aider pour le chargement des données

# Annexe - RAM et Mémoire virtuelle

