/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

declare module 'vue-echarts' {
  import { DefineComponent } from 'vue'
  const VChart: DefineComponent<any, any, any>
  export default VChart
}

