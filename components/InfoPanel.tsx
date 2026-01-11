import React from 'react';
import { TrainingStats, NetworkConfig, Neuron, Connection } from '../types';
import { Calculator, ArrowRight, Sigma, FunctionSquare, Info, Grid } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip } from 'recharts';

interface InfoPanelProps {
  stats: TrainingStats;
  config: NetworkConfig;
  selectedElement: { id: string; type: 'neuron' | 'connection' } | null;
  neurons: Neuron[];
  connections: Connection[];
}

const InfoPanel: React.FC<InfoPanelProps> = ({ stats, config, selectedElement, neurons, connections }) => {
  
  const renderNeuronDetails = (neuron: Neuron) => {
      // Find incoming connections
      const incoming = connections.filter(c => c.targetId === neuron.id);
      
      return (
          <div className="space-y-4">
              <div className="flex items-center gap-2 text-indigo-400 border-b border-slate-700 pb-2">
                  <Calculator className="w-4 h-4" />
                  <span className="font-bold text-sm">Neuron Calculation</span>
              </div>

              {/* 1. Linear Combination (Z) with Matrix View */}
              <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
                  <div className="flex items-center gap-2 mb-3 text-xs text-slate-400 font-mono">
                      <Sigma className="w-3 h-3" /> Linear Combination (Z)
                  </div>
                  
                  {incoming.length > 0 ? (
                      <>
                          {/* Matrix Vector Product Visualization */}
                          <div className="mb-4 overflow-x-auto pb-2 scrollbar-thin">
                            <div className="flex items-center justify-center gap-2 font-mono text-[10px]">
                                {/* Weights Column */}
                                <div className="flex flex-col items-center">
                                    <span className="text-[9px] text-slate-500 mb-1">Weights (W)</span>
                                    <div className="flex flex-col border-l-2 border-r-2 border-slate-600 px-2 py-1 gap-1 rounded-[4px] min-w-[3rem] items-center">
                                        {incoming.map(c => (
                                            <span key={c.id} className="text-emerald-400 whitespace-nowrap">
                                                {c.weight.toFixed(2)}
                                            </span>
                                        ))}
                                    </div>
                                </div>

                                <span className="text-slate-500 text-lg">×</span>

                                {/* Activations Column */}
                                <div className="flex flex-col items-center">
                                    <span className="text-[9px] text-slate-500 mb-1">Inputs (A)</span>
                                    <div className="flex flex-col border-l-2 border-r-2 border-slate-600 px-2 py-1 gap-1 rounded-[4px] min-w-[3rem] items-center">
                                        {incoming.map(c => {
                                            const src = neurons.find(n => n.id === c.sourceId);
                                            return (
                                                <span key={c.id} className="text-indigo-300 whitespace-nowrap">
                                                    {src?.activation.toFixed(2)}
                                                </span>
                                            );
                                        })}
                                    </div>
                                </div>

                                <span className="text-slate-500 text-lg">+</span>

                                {/* Bias */}
                                <div className="flex flex-col items-center">
                                    <span className="text-[9px] text-slate-500 mb-1">Bias (b)</span>
                                    <div className="flex flex-col border-l-2 border-r-2 border-slate-600 px-2 py-1 gap-1 rounded-[4px] min-h-[full] justify-center">
                                        <span className="text-amber-400">{neuron.bias.toFixed(2)}</span>
                                    </div>
                                </div>

                                <span className="text-slate-500 text-lg">=</span>

                                {/* Result */}
                                <div className="flex flex-col items-center">
                                    <span className="text-[9px] text-slate-500 mb-1">Z</span>
                                    <div className="flex flex-col border-l-2 border-r-2 border-white px-2 py-1 gap-1 rounded-[4px] bg-slate-900/50 shadow-lg shadow-indigo-500/10">
                                        <span className="text-white font-bold">{neuron.preActivation.toFixed(2)}</span>
                                    </div>
                                </div>
                            </div>
                          </div>

                          {/* Expanded Summation Details */}
                          <div className="space-y-1 font-mono text-[10px] text-slate-500 pt-2 border-t border-slate-700/50">
                             <div className="mb-1 text-[9px] uppercase tracking-wider text-slate-600">Detailed Summation</div>
                             {incoming.map(conn => {
                                 const sourceNeuron = neurons.find(n => n.id === conn.sourceId);
                                 return (
                                     <div key={conn.id} className="flex justify-between hover:bg-slate-700/30 rounded px-1 transition-colors">
                                         <span>
                                            <span className="text-emerald-500/70">w</span>
                                            <sub>{sourceNeuron?.neuronIndex}</sub>
                                            <span className="mx-1">·</span>
                                            <span className="text-indigo-400/70">a</span>
                                            <sub>{sourceNeuron?.neuronIndex}</sub>
                                         </span>
                                         <span>
                                             <span className="text-emerald-500">{conn.weight.toFixed(2)}</span>
                                             <span className="mx-1 text-slate-600">×</span>
                                             <span className="text-indigo-400">{sourceNeuron?.activation.toFixed(2)}</span>
                                         </span>
                                     </div>
                                 );
                             })}
                             <div className="flex justify-between px-1 text-amber-500/80 mt-1 border-t border-slate-700/30 pt-1">
                                 <span>+ bias</span>
                                 <span>{neuron.bias.toFixed(2)}</span>
                             </div>
                          </div>
                      </>
                  ) : (
                      <div className="text-xs text-slate-500 italic px-2">Input Layer (Raw Value)</div>
                  )}
              </div>

              {/* 2. Activation Function */}
              {incoming.length > 0 && (
                <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
                    <div className="flex items-center gap-2 mb-2 text-xs text-slate-400 font-mono">
                        <FunctionSquare className="w-3 h-3" /> Activation (a)
                    </div>
                    <div className="text-xs text-slate-300 mb-2">
                        Function: <span className="text-indigo-400 font-bold">{neuron.layerIndex === config.layers.length-1 ? config.layers[config.layers.length-1].activation : config.layers[1].activation}</span>
                    </div>
                    <div className="font-mono text-xs flex items-center justify-between text-white bg-slate-900/50 p-2 rounded border border-slate-700/50">
                        <span>f(z = {neuron.preActivation.toFixed(2)})</span>
                        <ArrowRight className="w-3 h-3 text-slate-600" />
                        <span className="text-base text-emerald-400 font-bold">{neuron.activation.toFixed(3)}</span>
                    </div>
                </div>
              )}

              {/* 3. Gradients (Backprop) */}
              <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
                  <div className="flex items-center justify-between mb-2 text-xs text-slate-400 font-mono">
                      <span>Gradient (δ)</span>
                      <span className={`font-bold ${neuron.gradient === 0 ? "text-slate-600" : "text-rose-400"}`}>
                          {neuron.gradient.toExponential(2)}
                      </span>
                  </div>
                  <p className="text-[10px] text-slate-500 leading-tight">
                      This value represents the partial derivative of the error with respect to this neuron's input.
                  </p>
              </div>
          </div>
      );
  };

  const renderConnectionDetails = (conn: Connection) => {
      return (
          <div className="space-y-4">
              <div className="flex items-center gap-2 text-indigo-400 border-b border-slate-700 pb-2">
                  <Calculator className="w-4 h-4" />
                  <span className="font-bold text-sm">Weight Update</span>
              </div>
              
              <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700 space-y-2">
                 <div className="flex justify-between text-xs font-mono">
                     <span className="text-slate-400">Current Weight</span>
                     <span className="text-white font-bold">{conn.weight.toFixed(4)}</span>
                 </div>
                 
                 <div className="flex justify-between text-xs font-mono">
                     <span className="text-slate-400">Gradient (dE/dw)</span>
                     <span className="text-rose-400">{conn.gradient.toExponential(3)}</span>
                 </div>

                 <div className="pt-2 border-t border-slate-600 mt-2">
                     <div className="text-[10px] text-slate-500 font-mono mb-1">Update Rule</div>
                     <div className="text-xs font-mono text-slate-300 bg-slate-900/50 p-2 rounded text-center">
                        w<sub>new</sub> = {conn.weight.toFixed(2)} - ({config.learningRate} × <span className="text-rose-400">{conn.gradient.toFixed(4)}</span>)
                     </div>
                 </div>
              </div>
          </div>
      );
  };

  return (
    <div className="absolute top-4 right-4 z-10 w-80 max-h-[calc(100vh-2rem)] flex flex-col gap-4">
        
        {/* Loss Chart */}
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-xl p-4 shadow-xl flex flex-col h-40">
            <div className="flex justify-between items-center mb-1">
                <h3 className="text-xs font-bold text-slate-400 uppercase">Training Loss</h3>
                <span className="text-xs font-mono text-indigo-400">Epoch {stats.epoch}</span>
            </div>
            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={stats.history}>
                        <XAxis dataKey="epoch" hide />
                        <YAxis domain={[0, 'auto']} hide />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '10px' }}
                            itemStyle={{ color: '#818cf8' }}
                        />
                        <Line type="monotone" dataKey="loss" stroke="#6366f1" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>

        {/* Math Inspector */}
        <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-xl p-4 shadow-xl flex-1 overflow-y-auto custom-scrollbar">
            {selectedElement ? (
                selectedElement.type === 'neuron' ? (
                    renderNeuronDetails(neurons.find(n => n.id === selectedElement.id)!)
                ) : (
                    renderConnectionDetails(connections.find(c => c.id === selectedElement.id)!)
                )
            ) : (
                <div className="h-full flex flex-col items-center justify-center text-slate-500 text-center p-4">
                    <Info className="w-8 h-8 mb-2 opacity-50" />
                    <p className="text-sm">Select a neuron or connection to inspect its internal calculations.</p>
                </div>
            )}
        </div>

    </div>
  );
};

export default InfoPanel;