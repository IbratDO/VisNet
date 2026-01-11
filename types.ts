export enum ActivationType {
  SIGMOID = 'SIGMOID',
  RELU = 'RELU',
  TANH = 'TANH'
}

export enum DatasetType {
  XOR = 'XOR',
  CIRCLE = 'CIRCLE',
  QUADRANTS = 'QUADRANTS'
}

export interface Neuron {
  id: string;
  layerIndex: number;
  neuronIndex: number;
  activation: number; // a (output)
  preActivation: number; // z (input sum + bias)
  bias: number;
  gradient: number; // delta (error term)
  position: [number, number, number];
}

export interface Connection {
  id: string;
  sourceId: string;
  targetId: string;
  weight: number;
  gradient: number; // gradient for weight update
  sourcePos: [number, number, number];
  targetPos: [number, number, number];
}

export interface LayerConfig {
  neurons: number;
  activation: ActivationType;
}

export interface NetworkConfig {
  learningRate: number;
  layers: LayerConfig[];
  dataset: DatasetType;
  batchSize: number;
}

export interface TrainingStats {
  epoch: number;
  loss: number;
  accuracy: number;
  currentInputIndex: number; // Track which data row is currently being processed
  history: { epoch: number; loss: number }[];
  batchProgress: number;
  batchSize: number;
}

export interface DataPoint {
  inputs: number[];
  target: number[];
}