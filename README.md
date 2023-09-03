# nvibe-rssi

### Reproduction des résultats

Afin d’exécuter le code, il faut se placer dans le dossier source et exécuter la commande bash suivante : 

```bash
python entrypoint.py --data-path data/classic_split --batch-size 16 --epochs 50 --hidden-size 128 --components 64 --output output --plot
```

L’analyse de données et la création des jeux de données se trouvent dans le notebook `data_analysis_rssi.ipynb`.
