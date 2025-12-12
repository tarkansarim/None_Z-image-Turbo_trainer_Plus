import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

// Job status constants
export type JobStatus = 'pending' | 'running' | 'stopped' | 'completed' | 'failed'

export interface Job {
  id: string
  name: string
  config_name: string
  status: JobStatus
  created_at: string
  updated_at: string
  started_at: string | null
  stopped_at: string | null
  completed_at: string | null
  current_epoch: number
  total_epochs: number
  current_step: number
  total_steps: number
  final_loss: number | null
  output_dir: string
  checkpoint_path: string | null
  lora_path: string | null
  error_message: string | null
  config_snapshot?: Record<string, any>
}

export interface JobLog {
  timestamp: string
  level: string
  message: string
}

export interface JobLossHistory {
  steps: number[]
  loss: (number | null)[]
  lr: (number | null)[]
}

export const useJobsStore = defineStore('jobs', () => {
  // State
  const jobs = ref<Job[]>([])
  const currentJob = ref<Job | null>(null)
  const runningJob = ref<Job | null>(null)
  const totalJobs = ref(0)
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Filters
  const statusFilter = ref<string>('all')
  const searchQuery = ref('')
  const page = ref(1)
  const pageSize = ref(20)

  // Computed
  const hasRunningJob = computed(() => runningJob.value !== null)

  const filteredJobs = computed(() => {
    return jobs.value
  })

  // Actions
  async function fetchJobs(options?: { status?: string; search?: string; limit?: number; offset?: number }) {
    loading.value = true
    error.value = null
    
    try {
      const params = new URLSearchParams()
      if (options?.status && options.status !== 'all') {
        params.append('status', options.status)
      }
      if (options?.search) {
        params.append('search', options.search)
      }
      params.append('limit', String(options?.limit || pageSize.value))
      params.append('offset', String(options?.offset || (page.value - 1) * pageSize.value))
      
      const response = await axios.get(`/api/jobs?${params.toString()}`)
      jobs.value = response.data.jobs
      totalJobs.value = response.data.total
    } catch (err: any) {
      error.value = err.response?.data?.detail || 'Failed to fetch jobs'
      console.error('Failed to fetch jobs:', err)
    } finally {
      loading.value = false
    }
  }

  async function fetchRunningJob() {
    try {
      const response = await axios.get('/api/jobs/running')
      runningJob.value = response.data.job
    } catch (err) {
      console.error('Failed to fetch running job:', err)
    }
  }

  async function fetchJob(jobId: string) {
    loading.value = true
    error.value = null
    
    try {
      const response = await axios.get(`/api/jobs/${jobId}`)
      currentJob.value = response.data
      return response.data
    } catch (err: any) {
      error.value = err.response?.data?.detail || 'Failed to fetch job'
      console.error('Failed to fetch job:', err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function createJob(name: string, configName: string, config: Record<string, any>, outputDir?: string) {
    loading.value = true
    error.value = null
    
    try {
      const response = await axios.post('/api/jobs', {
        name,
        config_name: configName,
        config,
        output_dir: outputDir
      })
      
      // Refresh job list
      await fetchJobs()
      
      return response.data.job_id
    } catch (err: any) {
      error.value = err.response?.data?.detail || 'Failed to create job'
      console.error('Failed to create job:', err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function updateJobConfig(jobId: string, config: Record<string, any>) {
    loading.value = true
    error.value = null
    
    try {
      await axios.put(`/api/jobs/${jobId}`, { config })
      
      // Refresh current job if it's the one we updated
      if (currentJob.value?.id === jobId) {
        await fetchJob(jobId)
      }
      
      // Refresh job list
      await fetchJobs()
      
      return true
    } catch (err: any) {
      error.value = err.response?.data?.detail || 'Failed to update job'
      console.error('Failed to update job:', err)
      return false
    } finally {
      loading.value = false
    }
  }

  async function resumeJob(jobId: string) {
    loading.value = true
    error.value = null
    
    try {
      const response = await axios.post(`/api/training/jobs/${jobId}/resume`)
      
      // Refresh running job status
      await fetchRunningJob()
      await fetchJobs()
      
      return response.data
    } catch (err: any) {
      error.value = err.response?.data?.detail || 'Failed to resume job'
      console.error('Failed to resume job:', err)
      throw err
    } finally {
      loading.value = false
    }
  }

  async function stopJob(jobId: string) {
    loading.value = true
    error.value = null
    
    try {
      await axios.post(`/api/training/jobs/${jobId}/stop`)
      
      runningJob.value = null
      await fetchJobs()
      
      return true
    } catch (err: any) {
      error.value = err.response?.data?.detail || 'Failed to stop job'
      console.error('Failed to stop job:', err)
      return false
    } finally {
      loading.value = false
    }
  }

  async function duplicateJob(jobId: string, newName?: string) {
    loading.value = true
    error.value = null
    
    try {
      const response = await axios.post(`/api/jobs/${jobId}/duplicate`, {
        new_name: newName
      })
      
      await fetchJobs()
      
      return response.data.job_id
    } catch (err: any) {
      error.value = err.response?.data?.detail || 'Failed to duplicate job'
      console.error('Failed to duplicate job:', err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function deleteJob(jobId: string, deleteOutputs: boolean = false) {
    loading.value = true
    error.value = null
    
    try {
      await axios.delete(`/api/jobs/${jobId}?delete_outputs=${deleteOutputs}`)
      
      // Clear current job if it was deleted
      if (currentJob.value?.id === jobId) {
        currentJob.value = null
      }
      
      await fetchJobs()
      
      return true
    } catch (err: any) {
      error.value = err.response?.data?.detail || 'Failed to delete job'
      console.error('Failed to delete job:', err)
      return false
    } finally {
      loading.value = false
    }
  }

  async function fetchJobLogs(jobId: string, limit: number = 500): Promise<JobLog[]> {
    try {
      const response = await axios.get(`/api/jobs/${jobId}/logs?limit=${limit}`)
      return response.data.logs
    } catch (err) {
      console.error('Failed to fetch job logs:', err)
      return []
    }
  }

  async function fetchJobLossHistory(jobId: string): Promise<JobLossHistory | null> {
    try {
      const response = await axios.get(`/api/jobs/${jobId}/history`)
      return response.data
    } catch (err) {
      console.error('Failed to fetch job loss history:', err)
      return null
    }
  }

  // Utility functions
  function getStatusColor(status: JobStatus): string {
    const colors: Record<JobStatus, string> = {
      pending: '#909399',    // gray
      running: '#409EFF',    // blue
      stopped: '#E6A23C',    // orange
      completed: '#67C23A',  // green
      failed: '#F56C6C'      // red
    }
    return colors[status] || '#909399'
  }

  function getStatusLabel(status: JobStatus): string {
    const labels: Record<JobStatus, string> = {
      pending: '待开始',
      running: '运行中',
      stopped: '已暂停',
      completed: '已完成',
      failed: '失败'
    }
    return labels[status] || status
  }

  function formatDate(dateStr: string | null): string {
    if (!dateStr) return '-'
    const date = new Date(dateStr)
    return date.toLocaleDateString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  function getProgressPercent(job: Job): number {
    if (!job.total_steps || job.total_steps === 0) return 0
    return Math.round((job.current_step / job.total_steps) * 100)
  }

  // Set filters
  function setStatusFilter(status: string) {
    statusFilter.value = status
    page.value = 1
    fetchJobs({ status, search: searchQuery.value })
  }

  function setSearchQuery(query: string) {
    searchQuery.value = query
    page.value = 1
    fetchJobs({ status: statusFilter.value, search: query })
  }

  function setPage(newPage: number) {
    page.value = newPage
    fetchJobs({
      status: statusFilter.value,
      search: searchQuery.value,
      offset: (newPage - 1) * pageSize.value
    })
  }

  return {
    // State
    jobs,
    currentJob,
    runningJob,
    totalJobs,
    loading,
    error,
    statusFilter,
    searchQuery,
    page,
    pageSize,
    
    // Computed
    hasRunningJob,
    filteredJobs,
    
    // Actions
    fetchJobs,
    fetchRunningJob,
    fetchJob,
    createJob,
    updateJobConfig,
    resumeJob,
    stopJob,
    duplicateJob,
    deleteJob,
    fetchJobLogs,
    fetchJobLossHistory,
    
    // Utilities
    getStatusColor,
    getStatusLabel,
    formatDate,
    getProgressPercent,
    
    // Filters
    setStatusFilter,
    setSearchQuery,
    setPage
  }
})

