import React from 'react';
import { Play, Pause, RotateCcw, Brain, Plus, Minus, FastForward, StepForward, Activity, Layers } from 'lucide-react';
import { NetworkConfig, ActivationType, DatasetType, DataPoint } from '../types';

interface ControlsProps {
  config: NetworkConfig;
  setConfig: (c: NetworkConfig) => void;
  isPlaying: boolean;
  speedMode: 'step' | 'fast';
  setSpeedMode: (m: 'step' | 'fast') => void;
  setIsPlaying: (p: boolean) => void;
  onStep: () => void;
  onReset: () => void;
  data: DataPoint[];
  currentInputIndex: number;
}

const Controls: React.FC<ControlsProps> = ({ config, setConfig, isPlaying, speedMode, setSpeedMode, setIsPlaying, onStep, onReset, data, currentInputIndex }) => {
  
  const handleLayerChange = (idx: number, neurons: number) => {
    const newLayers = [...config.layers];
    newLayers[idx] = { ...newLayers[idx], neurons: Math.max(1, Math.min(8, Number(neurons))) };
    setConfig({ ...config, layers: newLayers });
    onReset();
  };

  const addLayer = () => {
      const newLayers = [...config.layers];
      // Insert before output layer
      const outputLayer = newLayers.pop()!;
      newLayers.push({ neurons: 4, activation: ActivationType.TANH });
      newLayers.push(outputLayer);
      setConfig({ ...config, layers: newLayers });
      onReset();
  };

  const removeLayer = () => {
      if (config.layers.length <= 2) return; // Min Input + Output
      const newLayers = [...config.layers];
      newLayers.splice(newLayers.length - 2, 1); // Remove second to last
      setConfig({ ...config, layers: newLayers });
      onReset();
  };

  return (
    <div className="absolute top-4 left-4 z-10 w-80 flex flex-col gap-4 max-h-[calc(100vh-2rem)] overflow-y-auto pr-1">
      
      {/* Architecture Control */}
      <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-xl p-4 shadow-xl text-slate-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-bold flex items-center gap-2 uppercase tracking-wide text-slate-400">
            <Brain className="w-4 h-4 text-indigo-400" />
            Architecture
          </h2>
        </div>

        <div className="space-y-3">
            {/* Layer List */}
            <div className="space-y-2">
                {config.layers.map((layer, idx) => (
                    <div key={idx} className="flex items-center justify-between bg-slate-800 p-2 rounded">
                        <span className="text-xs font-mono text-slate-400">
                            {idx === 0 ? 'INPUT' : idx === config.layers.length - 1 ? 'OUTPUT' : `HIDDEN ${idx}`}
                        </span>
                        
                        {idx > 0 && idx < config.layers.length - 1 ? (
                             <div className="flex items-center gap-2">
                                <button onClick={() => handleLayerChange(idx, layer.neurons - 1)} className="p-1 hover:bg-slate-700 rounded"><Minus className="w-3 h-3" /></button>
                                <span className="text-sm w-4 text-center">{layer.neurons}</span>
                                <button onClick={() => handleLayerChange(idx, layer.neurons + 1)} className="p-1 hover:bg-slate-700 rounded"><Plus className="w-3 h-3" /></button>
                             </div>
                        ) : (
                            <span className="text-sm font-mono text-slate-500">{layer.neurons} Nodes</span>
                        )}
                    </div>
                ))}
            </div>

            <div className="flex gap-2 pt-2">
                <button onClick={addLayer} className="flex-1 bg-slate-800 hover:bg-slate-700 text-xs py-2 rounded flex items-center justify-center gap-1 transition-colors">
                    <Plus className="w-3 h-3" /> Add Layer
                </button>
                <button onClick={removeLayer} disabled={config.layers.length <= 2} className="flex-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-xs py-2 rounded flex items-center justify-center gap-1 transition-colors">
                    <Minus className="w-3 h-3" /> Remove Layer
                </button>
            </div>
            
            {/* Activation Config */}
            <div className="pt-2 border-t border-slate-700">
                <label className="text-xs text-slate-500 block mb-1">Hidden Activation</label>
                <select 
                    value={config.layers[1].activation}
                    onChange={(e) => {
                        const newLayers = config.layers.map((l, i) => 
                            (i > 0 && i < config.layers.length - 1) 
                            ? { ...l, activation: e.target.value as ActivationType } 
                            : l
                        );
                        setConfig({...config, layers: newLayers});
                        onReset();
                    }}
                    className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs"
                >
                    <option value={ActivationType.SIGMOID}>Sigmoid</option>
                    <option value={ActivationType.RELU}>ReLU</option>
                    <option value={ActivationType.TANH}>Tanh</option>
                </select>
            </div>
        </div>
      </div>

      {/* Playback Controls */}
      <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-xl p-4 shadow-xl">
         <h3 className="text-xs font-bold text-slate-400 uppercase mb-2 flex items-center gap-2">
             <Activity className="w-3 h-3" /> Training Control
         </h3>
         
         <div className="grid grid-cols-3 gap-2 mb-2">
             {/* Step Button */}
             <button
                onClick={onStep}
                className="flex flex-col items-center justify-center p-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors border border-slate-600"
             >
                 <StepForward className="w-5 h-5 mb-1 text-indigo-400" />
                 <span className="text-[10px] uppercase font-bold text-slate-300">Step</span>
             </button>

            {/* Fast Train Button */}
            <button
                onClick={() => {
                    setSpeedMode('fast');
                    setIsPlaying(!isPlaying);
                }}
                className={`flex flex-col items-center justify-center p-2 rounded-lg transition-colors border ${
                    isPlaying && speedMode === 'fast' 
                    ? 'bg-emerald-600/20 border-emerald-500 text-emerald-400' 
                    : 'bg-slate-700 hover:bg-slate-600 border-slate-600 text-slate-300'
                }`}
             >
                 {isPlaying && speedMode === 'fast' ? <Pause className="w-5 h-5 mb-1" /> : <FastForward className="w-5 h-5 mb-1 text-emerald-400" />}
                 <span className="text-[10px] uppercase font-bold">Fast</span>
             </button>

             {/* Reset Button */}
             <button
                onClick={onReset}
                className="flex flex-col items-center justify-center p-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors border border-slate-600"
             >
                 <RotateCcw className="w-5 h-5 mb-1 text-amber-400" />
                 <span className="text-[10px] uppercase font-bold text-slate-300">Reset</span>
             </button>
         </div>

        {/* Hyperparameters */}
        <div className="space-y-3 pt-1">
             <div>
                 <label className="text-xs text-slate-500 flex justify-between">
                    <span>Learning Rate</span>
                    <span>{config.learningRate}</span>
                 </label>
                 <input 
                    type="range" min="0.01" max="0.5" step="0.01"
                    value={config.learningRate}
                    onChange={(e) => setConfig({...config, learningRate: Number(e.target.value)})}
                    className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer mt-1"
                 />
             </div>

             <div className="flex justify-between items-center bg-slate-800 p-2 rounded border border-slate-700">
                <label className="text-xs text-slate-500 flex items-center gap-1">
                    <Layers className="w-3 h-3" /> Batch Size
                </label>
                <select 
                    value={config.batchSize}
                    onChange={(e) => setConfig({...config, batchSize: Number(e.target.value)})}
                    className="bg-slate-700 text-xs text-white border-none rounded px-2 py-1 outline-none cursor-pointer hover:bg-slate-600"
                >
                    <option value={1}>1 (SGD)</option>
                    <option value={2}>2</option>
                    <option value={4}>4</option>
                    <option value={8}>8 (Full)</option>
                </select>
             </div>
        </div>
      </div>

      {/* Data Table */}
      <div className="bg-slate-900/90 backdrop-blur-md border border-slate-700 rounded-xl p-4 shadow-xl text-slate-200 flex-1 flex flex-col min-h-[200px]">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-xs font-bold text-slate-400 uppercase">Input Data</h3>
            <select 
                value={config.dataset}
                onChange={(e) => {
                    setConfig({...config, dataset: e.target.value as DatasetType});
                    onReset();
                }}
                className="bg-slate-800 text-xs border border-slate-600 rounded px-1"
            >
                <option value={DatasetType.XOR}>XOR</option>
                <option value={DatasetType.CIRCLE}>Circle</option>
                <option value={DatasetType.QUADRANTS}>Linear</option>
            </select>
          </div>
          
          <div className="overflow-auto flex-1 border border-slate-800 rounded">
              <table className="w-full text-xs text-left">
                  <thead className="bg-slate-800 text-slate-400 sticky top-0">
                      <tr>
                          <th className="p-2">x1</th>
                          <th className="p-2">x2</th>
                          <th className="p-2">Target</th>
                      </tr>
                  </thead>
                  <tbody>
                      {data.map((row, i) => (
                          <tr key={i} className={`border-b border-slate-800 transition-colors duration-100 ${i === currentInputIndex ? 'bg-indigo-600 text-white font-bold' : 'text-slate-400'}`}>
                              <td className="p-2">{row.inputs[0].toFixed(2)}</td>
                              <td className="p-2">{row.inputs[1].toFixed(2)}</td>
                              <td className="p-2">{row.target[0]}</td>
                          </tr>
                      ))}
                  </tbody>
              </table>
          </div>
      </div>

    </div>
  );
};

export default Controls;