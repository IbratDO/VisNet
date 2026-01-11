import { ActivationType, DatasetType, LayerConfig, Neuron, Connection } from '../types';

// Mathematical functions
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const sigmoidDerivative = (x: number) => x * (1 - x);

const relu = (x: number) => Math.max(0, x);
const reluDerivative = (x: number) => (x > 0 ? 1 : 0);

const tanh = (x: number) => Math.tanh(x);
const tanhDerivative = (x: number) => 1 - (x * x);

export class MiniNeuralNet {
  layers: LayerConfig[];
  weights: number[][][]; // [layer][from][to]
  biases: number[][];    // [layer][neuron]
  
  // State for visualization
  activations: number[][]; // [layer][neuron] (a)
  preActivations: number[][]; // [layer][neuron] (z)
  neuronGradients: number[][]; // [layer][neuron] (delta)
  weightGradients: number[][][]; // [layer][from][to]
  
  learningRate: number;
  batchSize: number;
  currentBatchCount: number = 0;

  // Accumulators for Batching
  accWeightGradients: number[][][];
  accBiasGradients: number[][];

  constructor(layers: LayerConfig[], learningRate: number = 0.1, batchSize: number = 1) {
    this.layers = layers;
    this.learningRate = learningRate;
    this.batchSize = batchSize;
    this.weights = [];
    this.biases = [];
    
    // Init state arrays
    this.activations = [];
    this.preActivations = [];
    this.neuronGradients = [];
    this.weightGradients = [];
    
    // Init accumulators placeholder
    this.accWeightGradients = [];
    this.accBiasGradients = [];

    this.initializeWeights();
  }

  public updateConfig(learningRate: number, batchSize: number) {
      this.learningRate = learningRate;
      if (this.batchSize !== batchSize) {
          this.batchSize = batchSize;
          this.currentBatchCount = 0;
          this.resetAccumulators();
      }
  }

  private resetAccumulators() {
    this.accWeightGradients = this.weights.map(l => l.map(n => new Array(n.length).fill(0)));
    this.accBiasGradients = this.biases.map(l => new Array(l.length).fill(0));
    this.currentBatchCount = 0;
  }

  private initializeWeights() {
    // Initialize weights: Random Uniform [-1, 1]
    // Initialize biases: Random Uniform [-0.5, 0.5]
    for (let i = 1; i < this.layers.length; i++) {
      const prevNeurons = this.layers[i - 1].neurons;
      const currNeurons = this.layers[i].neurons;
      
      const layerWeights: number[][] = [];
      const layerWeightGrads: number[][] = [];

      for (let j = 0; j < prevNeurons; j++) {
        const neuronWeights: number[] = [];
        const neuronWeightGrads: number[] = [];
        for (let k = 0; k < currNeurons; k++) {
          neuronWeights.push(Math.random() * 2 - 1); 
          neuronWeightGrads.push(0);
        }
        layerWeights.push(neuronWeights);
        layerWeightGrads.push(neuronWeightGrads);
      }
      this.weights.push(layerWeights);
      this.weightGradients.push(layerWeightGrads);

      const layerBiases: number[] = [];
      const layerNeuronGrads: number[] = [];
      for (let k = 0; k < currNeurons; k++) {
        layerBiases.push(Math.random() - 0.5);
        layerNeuronGrads.push(0);
      }
      this.biases.push(layerBiases);
      this.neuronGradients.push(layerNeuronGrads);
    }
    
    // Init Input layer grads holder (unused but keeps indexing consistent)
    this.neuronGradients.unshift(new Array(this.layers[0].neurons).fill(0));

    // Initialize accumulators based on the structure we just built
    this.resetAccumulators();
  }

  public forward(inputs: number[]): number[] {
    this.activations = [inputs];
    this.preActivations = [inputs]; // Input layer z = a

    let currentActivations = inputs;

    for (let i = 0; i < this.weights.length; i++) {
      const nextActivations: number[] = [];
      const nextPreActivations: number[] = [];
      
      const layerConfig = this.layers[i + 1];
      const layerWeights = this.weights[i];
      const layerBiases = this.biases[i];

      for (let j = 0; j < layerConfig.neurons; j++) {
        // z = Sum(w * a_prev) + b
        let sum = layerBiases[j];
        for (let k = 0; k < currentActivations.length; k++) {
          sum += currentActivations[k] * layerWeights[k][j];
        }
        nextPreActivations.push(sum);

        // a = Activation(z)
        let val = 0;
        switch (layerConfig.activation) {
          case ActivationType.SIGMOID: val = sigmoid(sum); break;
          case ActivationType.RELU: val = relu(sum); break;
          case ActivationType.TANH: val = tanh(sum); break;
        }
        nextActivations.push(val);
      }
      
      this.activations.push(nextActivations);
      this.preActivations.push(nextPreActivations);
      currentActivations = nextActivations;
    }

    return currentActivations;
  }

  public train(inputs: number[], targets: number[]) {
    // 1. Forward
    const outputs = this.forward(inputs);

    // Reset current gradients structure (per sample)
    this.neuronGradients = this.layers.map(l => new Array(l.neurons).fill(0));
    this.weightGradients = this.weights.map(l => l.map(n => new Array(n.length).fill(0)));

    // 2. Output Error (MSE Derivative)
    const outputLayerIdx = this.layers.length - 1;
    for (let i = 0; i < outputs.length; i++) {
      const error = outputs[i] - targets[i]; 
      let activationPrime = 0;
      const z = this.preActivations[outputLayerIdx][i]; 
      
      switch (this.layers[outputLayerIdx].activation) {
          case ActivationType.SIGMOID: activationPrime = sigmoidDerivative(outputs[i]); break;
          case ActivationType.RELU: activationPrime = reluDerivative(z); break;
          case ActivationType.TANH: activationPrime = tanhDerivative(outputs[i]); break;
      }
      
      this.neuronGradients[outputLayerIdx][i] = error * activationPrime;
    }

    // 3. Backpropagate & Accumulate
    for (let i = this.weights.length - 1; i >= 0; i--) {
        const prevLayerActivations = this.activations[i];
        const nextLayerGradients = this.neuronGradients[i + 1];
        
        // Calculate Weight Gradients
        for (let j = 0; j < this.weights[i].length; j++) { 
            for (let k = 0; k < this.weights[i][j].length; k++) { 
                // dE/dw = delta_next * a_prev
                const gradient = nextLayerGradients[k] * prevLayerActivations[j];
                this.weightGradients[i][j][k] = gradient;
                
                // ACCUMULATE (Don't update yet)
                this.accWeightGradients[i][j][k] += gradient;
            }
        }

        // Accumulate Biases
        for (let k = 0; k < this.biases[i].length; k++) {
            this.accBiasGradients[i][k] += nextLayerGradients[k];
        }

        // Calculate Gradients for Previous Layer (delta)
        if (i > 0) {
            for (let j = 0; j < this.layers[i].neurons; j++) {
                let errorSum = 0;
                for (let k = 0; k < this.layers[i+1].neurons; k++) {
                    errorSum += nextLayerGradients[k] * this.weights[i][j][k];
                }

                let activationPrime = 0;
                const z = this.preActivations[i][j];
                const a = this.activations[i][j];

                switch (this.layers[i].activation) {
                    case ActivationType.SIGMOID: activationPrime = sigmoidDerivative(a); break;
                    case ActivationType.RELU: activationPrime = reluDerivative(z); break;
                    case ActivationType.TANH: activationPrime = tanhDerivative(a); break;
                }
                this.neuronGradients[i][j] = errorSum * activationPrime;
            }
        }
    }
    
    this.currentBatchCount++;

    // 4. Check Batch & Update
    if (this.currentBatchCount >= this.batchSize) {
        // Average the gradients
        const scaler = 1 / this.batchSize;
        
        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    const avgGrad = this.accWeightGradients[i][j][k] * scaler;
                    this.weights[i][j][k] -= this.learningRate * avgGrad;
                }
            }
            for (let k = 0; k < this.biases[i].length; k++) {
                const avgGrad = this.accBiasGradients[i][k] * scaler;
                this.biases[i][k] -= this.learningRate * avgGrad;
            }
        }
        
        // Reset Accumulators
        this.resetAccumulators();
    }

    return outputs;
  }

  public getVisualizationData() {
    const neurons: Neuron[] = [];
    const connections: Connection[] = [];

    const layerSpacing = 6;
    const neuronSpacing = 2;

    for (let i = 0; i < this.layers.length; i++) {
      const layerCount = this.layers[i].neurons;
      const x = (i - (this.layers.length - 1) / 2) * layerSpacing;
      
      for (let j = 0; j < layerCount; j++) {
        const y = (j - (layerCount - 1) / 2) * neuronSpacing;
        const id = `l${i}-n${j}`;
        
        neurons.push({
          id,
          layerIndex: i,
          neuronIndex: j,
          activation: this.activations[i] ? this.activations[i][j] : 0,
          preActivation: this.preActivations[i] ? this.preActivations[i][j] : 0,
          bias: i > 0 ? this.biases[i-1][j] : 0,
          gradient: this.neuronGradients[i] ? this.neuronGradients[i][j] : 0,
          position: [x, y, 0]
        });
      }
    }

    for (let i = 0; i < this.weights.length; i++) {
      const sourceLayerIdx = i;
      const targetLayerIdx = i + 1;
      
      const sourceNeurons = neurons.filter(n => n.layerIndex === sourceLayerIdx);
      const targetNeurons = neurons.filter(n => n.layerIndex === targetLayerIdx);

      for (let j = 0; j < sourceNeurons.length; j++) {
        for (let k = 0; k < targetNeurons.length; k++) {
           const weight = this.weights[i][j][k];
           const grad = this.weightGradients[i] ? this.weightGradients[i][j][k] : 0;
           
           connections.push({
             id: `c-l${i}n${j}-l${targetLayerIdx}n${k}`,
             sourceId: sourceNeurons[j].id,
             targetId: targetNeurons[k].id,
             weight: weight,
             gradient: grad,
             sourcePos: sourceNeurons[j].position,
             targetPos: targetNeurons[k].position
           });
        }
      }
    }

    return { neurons, connections };
  }
}

export const generateData = (type: DatasetType, count: number = 8) => {
  const data: { inputs: number[], target: number[] }[] = [];
  
  if (type === DatasetType.XOR) {
      data.push({ inputs: [0.1, 0.1], target: [0] });
      data.push({ inputs: [0.9, 0.1], target: [1] });
      data.push({ inputs: [0.1, 0.9], target: [1] });
      data.push({ inputs: [0.9, 0.9], target: [0] });
      
      data.push({ inputs: [0.2, 0.2], target: [0] });
      data.push({ inputs: [0.8, 0.2], target: [1] });
      data.push({ inputs: [0.2, 0.8], target: [1] });
      data.push({ inputs: [0.8, 0.8], target: [0] });
  } else if (type === DatasetType.QUADRANTS) {
      for(let i=0; i<8; i++) {
        const x = (i % 2 === 0 ? -1 : 1) * (0.5 + Math.random()*0.4);
        const y = Math.random()*2 - 1;
        data.push({ inputs: [x, y], target: [x > 0 ? 1 : 0]});
      }
  } else {
      // Circle
      for(let i=0; i<8; i++) {
          const angle = (i / 8) * Math.PI * 2;
          const r = i % 2 === 0 ? 0.3 : 0.9;
          const x = Math.cos(angle) * r;
          const y = Math.sin(angle) * r;
          data.push({ inputs: [x, y], target: [r < 0.5 ? 1 : 0]});
      }
  }
  return data;
};