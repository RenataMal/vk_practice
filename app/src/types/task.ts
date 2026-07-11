export type TaskState =
  | 'queued'
  | 'decoding'
  | 'analyzing'
  | 'enhancing'
  | 'encoding'
  | 'completed'
  | 'cancelled'
  | 'failed';

export interface TaskStatus {
  taskId: string;
  status: TaskState;
  progress: number;
  error: string | null;
}

export interface EnhancementParameters {
  brightness: number;
  contrast: number;
  saturation: number;
}

export interface ProcessingMetrics {
  width: number;
  height: number;
  megapixels: number;
  decodeMs: number;
  analysisMs: number;
  enhancementMs: number;
  encodingMs: number;
  totalMs: number;
}

export interface EnhancementResult {
  blob: Blob;
  parameters: EnhancementParameters;
  metrics: ProcessingMetrics;
}

export interface ProcessImageMessage {
  type: 'process';
  taskId: string;
  file: File;
}

export interface WorkerStatusMessage {
  type: 'status';
  taskId: string;
  status: TaskState;
  progress: number;
}

export interface WorkerResultMessage {
  type: 'result';
  taskId: string;
  result: EnhancementResult;
}

export interface WorkerErrorMessage {
  type: 'error';
  taskId: string;
  error: string;
}

export type WorkerInputMessage = ProcessImageMessage;

export type WorkerOutputMessage =
  | WorkerStatusMessage
  | WorkerResultMessage
  | WorkerErrorMessage;
