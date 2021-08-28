from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


class EpochMetrics:
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.current_epoch = 1
        self.batch_counter = 0
        self.history = {  }
        
    def start_new_epoch(self):
        
        self.history["epoch" + str(self.current_epoch)] = self.metrics.get_batch_metrics(self.batch_counter)
        self.current_epoch += 1
        self.batch_counter = 0
        self.metrics.reset_metrics()
        
    def update_metrics(self, loss, outputs, labels):
        
        self.metrics.update_metrics(loss, outputs, labels)
        self.batch_counter += 1


class MetricsAccumulator:
    
    def __init__(self):
        
        self.loss = []
        self.acc = []
        self.f1 = []
        self.auc = []
        self.precision = []
        self.recall = []
        
        
    def update_metrics(self, loss, outputs, labels):
        
        self.loss.append(loss)
        self.acc.append(accuracy_score(outputs, labels))
        self.f1.append(f1_score(labels, outputs, average='weighted'))
        self.precision.append(precision_score(labels, outputs, average='weighted'))
        self.recall.append(recall_score(labels, outputs, average='weighted'))
        self.auc.append(roc_auc_score(labels, outputs))
        
    def get_accumulated_metrics(self, num_batches: int):
        
        return { "loss" : self.loss, "acc" : self.acc, "f1" : self.f1, "auc" : self.auc, 
                 "precision" : self.precision, "recall" : self.recall}


class PerformanceMetrics:
    
    def __init__(self):
        
        self.running_loss = 0.0
        self.running_acc = 0.0
        self.running_f1 = 0.0
        self.running_auc = 0.0
        self.running_precision = 0.0
        self.running_recall = 0.0

    def update_metrics(self, loss, outputs, labels):
        
        self.running_loss += loss
        self.running_acc += accuracy_score(outputs, labels)
        self.running_f1 += f1_score(labels, outputs, average='weighted')
        self.running_precision += precision_score(labels, outputs, average='weighted')
        self.running_recall += recall_score(labels, outputs, average='weighted')
        self.running_auc += roc_auc_score(labels, outputs)
        
        
    def get_batch_metrics(self, num_batches: int):
        
        return { "loss" : self.running_loss / num_batches, "acc" : self.running_acc / num_batches,
                "f1" : self.running_f1 / num_batches, "auc" : self.running_auc / num_batches, 
                "precision" : self.running_precision / num_batches , "recall" : self.running_recall / num_batches }
    
    def to_string(self, num_batches: int):
        
        metrics = self.get_batch_metrics(num_batches)
        return f'loss: {metrics["loss"]:.4f} | acc: {metrics["acc"]:.4f} | auc: {metrics["auc"]:.4f} | precision: {metrics["precision"]:.4f} | recall: {metrics["recall"]:.4f} | f1: {metrics["f1"]:.4f}' 
        
    def reset_metrics(self):
        
        self.running_loss = 0.0
        self.running_acc = 0.0
        self.running_f1 = 0.0
        self.running_auc = 0.0
        self.running_precision = 0.0
        self.running_recall = 0.0