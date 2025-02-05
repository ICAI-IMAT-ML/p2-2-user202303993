# Laboratory practice 2.2: KNN classification
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import numpy as np  
import seaborn as sns

 

def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """
 

    return  np.sum(np.abs(a - b) ** p) ** (1 / p)


# k-Nearest Neighbors Model

# - [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html#classification)
# - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        # TODO
        # Validaciones
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Length of X_train and y_train must be equal.")
        if not isinstance(k, int) or k <= 0 or not isinstance(p, int) or p <= 0:
            raise ValueError("k and p must be positive integers.")
         
        self.k = k
        self.p = p
        self.x_train = X_train
        self.y_train = y_train

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        # TODO
        predictions = []

        for x_test in X:
            #para cada punto de  los datos de test calculamos su distancia minkowski con cada uno de los puntos de de de los datos de entrenamiento 
            np_array_distancias = self.compute_distances(x_test)
            
            #del array distances que acabamos de crear, argosrt devuelve los indices del array ordenado 
            k_cercanos_distancias = self.get_k_nearest_neighbors(np_array_distancias)
            
            #obtener las etiquetas de los k vecinos más cercanos al acceder a los datos de entrenamiento de estos k vecinos
            k_cercanos_etiquetas = self.y_train[k_cercanos_distancias]
            
            #determinamos la clase más común entre los vecinos
            k_cercanos_etiquetas = np.array([1, 0, 0]) #convertimos lista en array
            mas_comun = self.most_common_label(k_cercanos_etiquetas)
            predictions.append(mas_comun)

        return np.array(predictions)
        
    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        # TODO
        probabilities = []

        for x_test in X:
            #para cada punto de  los datos de test calculamos su distancia minkowski con cada uno de los puntos de de de los datos de entrenamiento 
            np_array_distancias = self.compute_distances(x_test)
            
            #del array distances que acabamos de crear, argosrt devuelve los indices del array ordenado 
            k_cercanos_distancias = self.get_k_nearest_neighbors(np_array_distancias)
            
            #obtener las etiquetas de los k vecinos más cercanos al acceder a los datos de entrenamiento de estos k vecinos
            k_cercanos_etiquetas = self.y_train[k_cercanos_distancias]
            
            #determinamos la clase más común entre los vecinos
            k_cercanos_etiquetas = np.array([1, 0, 0]) #convertimos lista en array
            etiquetas, counts = np.unique(k_cercanos_etiquetas, return_counts=True) #devuelve las etiquetas que hay (en este caso 1 y 0 ) y un recuento de cuantas veces aparecen para los k vecinos
            prob = counts / self.k
            probabilities.append(prob)
            
        return np.array(probabilities)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        # TODO

        distances = np.array([minkowski_distance(point, train_point, self.p) for train_point in self.x_train])
        return distances


    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """
        # TODO
        k_cercanos_distancias = np.argsort(distances)[:self.k]  #slicing para quedarnos solo con los k mas cercanos
        return k_cercanos_distancias

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        # TODO
        etiquetas, counts = np.unique(knn_labels, return_counts=True) #devuelve las etiquetas que hay (en este caso 1 y 0 ) y un recuento de cuantas veces aparecen para los k vecinos
        mas_comun = etiquetas[np.argmax(counts)] 
        return mas_comun

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"



def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.

    This function creates two plots:
    1. A classification results plot showing True Positives, False Positives, False Negatives, and True Negatives.
    2. A predicted probabilities plot showing the probability predictions with level curves for each 0.1 increment.

    Args:
        X (np.ndarray): The input data, a 2D array of shape (n_samples, 2), where each row represents a sample and each column represents a feature.
        y (np.ndarray): The true labels, a 1D array of length n_samples.
        model (classifier): A trained classification model with 'predict' and 'predict_proba' methods. The model should be compatible with the input data 'X'.
        grid_points_n (int): The number of points in the grid along each axis. This determines the resolution of the plots.

    Returns:
        None: This function does not return any value. It displays two plots.

    Note:
        - This function assumes binary classification and that the model's 'predict_proba' method returns probabilities for the positive class in the second column.
    """
    # Map string labels to numeric
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Predict on input data
    preds = model.predict(X)

    # Determine TP, FP, FN, TN
    tp = (y == unique_labels[1]) & (preds == unique_labels[1])
    fp = (y == unique_labels[0]) & (preds == unique_labels[1])
    fn = (y == unique_labels[1]) & (preds == unique_labels[0])
    tn = (y == unique_labels[0]) & (preds == unique_labels[0])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Classification Results Plot
    ax[0].scatter(X[tp, 0], X[tp, 1], color="green", label=f"True {num_to_label[1]}")
    ax[0].scatter(X[fp, 0], X[fp, 1], color="red", label=f"False {num_to_label[1]}")
    ax[0].scatter(X[fn, 0], X[fn, 1], color="blue", label=f"False {num_to_label[0]}")
    ax[0].scatter(X[tn, 0], X[tn, 1], color="orange", label=f"True {num_to_label[0]}")
    ax[0].set_title("Classification Results")
    ax[0].legend()

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points_n),
        np.linspace(y_min, y_max, grid_points_n),
    )

    # # Predict on mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Use Seaborn for the scatter plot
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", ax=ax[1])
    ax[1].set_title("Classes and Estimated Probability Contour Lines")

    # Plot contour lines for probabilities
    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors="black")
    ax[1].clabel(cnt, inline=True, fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()



def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix components
    tp = np.sum((y_true_mapped == 1) & (y_pred_mapped == 1))
    fp = np.sum((y_true_mapped == 0) & (y_pred_mapped == 1))
    fn = np.sum((y_true_mapped == 1) & (y_pred_mapped == 0))
    tn = np.sum((y_true_mapped == 0) & (y_pred_mapped == 0))

    #formula de Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    #formula de Precision
    precision = tp / (tp + fp) if tp + fp != 0 else 0

    #fromula de Recall (Sensitivity)
    recall = tp / (tp + fn) if tp + fn != 0 else 0

    #formula de Specificity
    specificity = tn / (tn + fp) if tn + fp != 0 else 0

    #fromula de F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return {
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }
    


def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin

    """
    # TODO
    #Bin de las probabilidades predichas
    bins = np.linspace(0, 1, n_bins + 1)  #creamos n_bins + 1 límites para las probabilidades (divide el rango de 0 a 1 en intervalos iguales.)
    bin_centers = (bins[:-1] + bins[1:]) / 2  #calculamos los centros de los bins

    #proporciones verdaderas para cada bin:
    true_proportions = []
    for i in range(n_bins):
        #seleccionamos las muestras cuyo valor de probabilidad se encuentra dentro del rango del bin actual
        bin_mask = (y_probs >= bins[i]) & (y_probs < bins[i + 1])
        
        #calculamos la proporción de resultados positivos dentro de ese bin
        true_positive_count = np.sum(y_true[bin_mask] == positive_label)
        total_in_bin = np.sum(bin_mask)
        #calculamos la proporción de verdaderos positivos dentro del bin
        true_proportion = true_positive_count / total_in_bin if total_in_bin > 0 else 0
        true_proportions.append(true_proportion)

    return {"bin_centers": bin_centers, "true_proportions": true_proportions}



def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10. 
                                Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "array_passed_to_histogram_of_positive_class": 
                Array of predicted probabilities for the positive class.
            - "array_passed_to_histogram_of_negative_class": 
                Array of predicted probabilities for the negative class.

    """
    # TODO
    #mapeamos las etiquetas verdaderas a formato binario:
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    return {
        "array_passed_to_histogram_of_positive_class": y_probs[y_true_mapped == 1],
        "array_passed_to_histogram_of_negative_class": y_probs[y_true_mapped == 0],
    }



def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.

    Returns:
        dict: A dictionary containing the following:
            - "fpr": Array of False Positive Rates for each threshold.
            - "tpr": Array of True Positive Rates for each threshold.

    """
    # TODO
    #Mapeamos las etiquetas verdaderas a etiquetas binarias ,1 para positiva, 0 para negativa
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])

    #Creamos un array de umbrales desde 0 hasta 1 con un paso de 0.1
    thresholds = np.linspace(0, 1, 11)

    #inicializamos listas vacías para almacenar FPR y TPR
    fpr = []
    tpr = []

    #calculamos TPR y FPR para cada umbral
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)  
        
        tp = np.sum((y_true_mapped == 1) & (y_pred == 1))
        fp = np.sum((y_true_mapped == 0) & (y_pred == 1))
        fn = np.sum((y_true_mapped == 1) & (y_pred == 0))
        tn = np.sum((y_true_mapped == 0) & (y_pred == 0))

        tpr_value = tp / (tp + fn) if tp + fn != 0 else 0
        fpr_value = fp / (fp + tn) if fp + tn != 0 else 0

        tpr.append(tpr_value)
        fpr.append(fpr_value)

    return {"fpr": np.array(fpr), "tpr": np.array(tpr)}
