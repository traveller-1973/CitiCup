// main.ts
import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import Echarts from "vue-echarts"
import * as echarts from "echarts"

const app = createApp(App)

app.use(ElementPlus)
app.mount('#app')
app.component("v-chart", Echarts)
app.config.globalProperties.$echarts = echarts