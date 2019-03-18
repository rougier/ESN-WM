
## Idées d'expériences à faire

### Reservoir working memory : add noise in output
Tester d'ajouter du bruit dans la sortie seulement (pendant l'apprentissage et-ou durant le test) au lieu de mettre du bruit dans l'activité des neurones. L'idée c'est que l'ajout de bruit dans les neurones de sortie peut améliorer le fait que le réseau se comporte comme un attracteur, et ainsi revienne vers la valeur où il était avant s'il a été perturbé dans une direction orthogonale au line attractor.

### Exp line-attractor: step1:
Stabiliser l'apprentissage d'une valeur avec de la redondance. i.e. réimplémenter la 3-gated task où le réseau est entrainé à sortir la même valeur sur les 3 sorties et en ajoutant du bruit durant l'entrainement seulement sur les sorties. (peut être cela nécessite un apprentissage online, mais avec du teacher forcing où on ajoute du bruit ça devrait pouvoir marcher aussi.

### Exp line-attractor: step2:
comparer un 1-gate-net avec un 3-gate-net-1-robust-value en comparant leurs réécritures afin de voir si "c'est les mêmes réseaux au final" comme le prédit Anthony. Xavier prédit que comme les 3-gate-net aura appris a être plus robuste cela permettra aux poids appris d'avoir une propriété particulière pour la stabilité et la correction de "drifts" de la valeur gardée en mémoire et qui arrivent au cours du temps.


--- (en dessous : autres analyses à garder en tête)


\tdg{Faire la figure de "robusness of the model to hyperparameters" pour la 3-gated task.}

\tdg{Faire les scatterplot des poids Wout, W_{in}_{T} et W_{fb} pour voir si les poids qui sont en haut à droite du graphique Wout vs Wfb sont ceux qui sont le plus corrélés avec la sortie ; idem pour Win vs Wfb vs Wout (en 3D).}

\tdg{Afin de comprendre pourquoi dans la fig 5 (robustess param) l'erreur stagne à partir de 7 "training values", faire la réécriture à chaque fois qu'une valeur est apprise et regarder l'évolution du spectre des valeurs propres -> mon hypothèse est que le spectre des valeurs propres évolue jusqu'à 7 valeurs apprises et qu'ensuite il reste fixe. -> IDEM pour avec un apprentissage online (ex RLS) afin de voir quand est-ce que les poids "arrêtent" de beaucoup changer, et si on trouve aussi cette valeur 7.}
