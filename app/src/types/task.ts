export type TaskState =
  | 'queued'
  | 'decoding'
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
  blob: Blob;
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
