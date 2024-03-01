<template>
  <div>
    <!-- 顶部导航栏 -->
    <el-menu mode="horizontal" background-color="#324157" text-color="#fff" active-text-color="#ffd04b">
      <el-menu-item index="1">首页</el-menu-item>
      <el-menu-item index="2">策略研究</el-menu-item>
      <el-menu-item index="3">数据字典</el-menu-item>
      <el-menu-item index="4">社区</el-menu-item>
      <el-menu-item index="5">帮助</el-menu-item>
      <el-menu-item index="6">聚宽投资</el-menu-item>
      <el-menu-item index="7">本地数据</el-menu-item>
      <el-menu-item index="8">聚宽会员</el-menu-item>
    </el-menu>

    <!-- 因子名称和代码的标题 -->
    <div class="factor-title">
      <h2>净利润与营业总收入之比 net_profit_to_total_operate_revenue_ttm</h2>
      <el-button type="primary">获取因子值</el-button>
    </div>

    <!-- 因子详细信息表格 -->
    <el-table :data="tableData" style="width: 100%">
      <el-table-column prop="attribute" width="180"></el-table-column>
      <el-table-column prop="value" ></el-table-column>
    </el-table>

    <!-- 回测参数设置 -->
    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="6" v-for="(param, index) in backtestParams" :key="index">
        <div v-if="param.type !== 'button'" class="param-item">
          <span class="param-name">{{ param.name }}：</span>
          <el-select v-model="param.value" placeholder="请选择" class="param-select">
            <el-option v-for="option in param.options" :key="option.value" :label="option.label" :value="option.value">
            </el-option>
          </el-select>
        </div>
        <div v-else class="button-container">
          <el-button type="primary" @click="startBacktest">确定</el-button>
        </div>
      </el-col>
    </el-row>

    <!-- 指标表格 -->
    <div class="indicator-table">
      <div class="indicator-row">
        <div class="indicator-name">IC均值</div>
        <div class="indicator-name">|IC|>0.02的比率</div>
        <div class="indicator-name">IR值</div>
      </div>
      <div class="indicator-row">
        <div class="indicator-value"><h1>0.000</h1></div>
        <div class="indicator-value"><h1>0.816</h1></div>
        <div class="indicator-value"><h1>0.003</h1></div>
      </div>
    </div>

    <!-- 因子详情组合图表 -->
    <div class="factor-details">
      <!-- 表格 -->
      <el-table :data="factorDetailsTableData" style="width: 100%">
        <el-table-column prop="data1"  align="center"></el-table-column>
        <el-table-column prop="data2"  align="center"></el-table-column>
        <el-table-column prop="data3"  align="center"></el-table-column>
        <el-table-column prop="data4"  align="center"></el-table-column>
        <el-table-column prop="data5"  align="center"></el-table-column>
        <el-table-column prop="data6"  align="center"></el-table-column>
        <el-table-column prop="data7"  align="center"></el-table-column>
        
      </el-table>

      <!-- 选项栏 -->
      <div class="options-bar">
        <el-radio-group v-model="yAxisType">
          <el-radio label="normal">普通轴</el-radio>
          <el-radio label="logarithmic">对数轴</el-radio>
        </el-radio-group>

        <el-checkbox v-model="showAllQuantiles">全分位</el-checkbox>
      </div>

      <!-- 折线图 因子详情 -->
      <div ref="factorDetailsChart" style="width: 100%; height: 400px;"></div>

    <div class="chart-row">
      <!-- 折线图 IC时序图-->
      <div ref="ICTimeSeriesChart" style="width: 50%; height: 400px;"></div>

      <!-- 柱形图 行业IC-->
      <div ref="industryICChart" style="width: 50%; height: 400px;"></div>
    </div>

    </div>
   <!-- HTML -->
<div class="footer">
  <div class="container">
    <div class="row">
      <div class="col-md-4">
        <h4>关于</h4>
        <p>聚宽是一个量化投资交易服务平台，致力于降低宽客的门槛，打造最高效、易用的量化交易平台。</p>
      </div>
      <div class="col-md-4">
        <h4>帮助</h4>
        <ul>
          <div class="dropdown-content">
  <a href="^9^">公司介绍</a>
  <a href="^10^">团队介绍</a>
  <a href="^11^">联系我们</a>
  <a href="^12^">加入我们</a>
</div>
        </ul>
      </div>
      <div class="col-md-4">
        <h4>联系我们</h4>
        <p>电话：400-666-5105</p>
        <p>邮箱：hi@joinquant.com</p>
        <p>地址：北京市海淀区中关村东路1号院5号楼</p>
      </div>
    </div>
    <div class="row">
      <div class="col-md-12">
        <p class="copy">©2024 @joinquant.com | 京ICP备17068639号-5 | 增值电信业务经营许可证：京B2-20212305</p>
      </div>
    </div>
  </div>
</div>

  </div>

</template>


<script setup>
import { ref } from "vue";
import { ElMessage } from 'element-plus';
import axios from 'axios';

const tableData = ref([
  { attribute: '因子类别', value: '质量类因子' },
  { attribute: '计算公式', value: '净利润与营业总收入之比=净利润（TTM）/营业总收入（TTM）' },
  { attribute: '更新时间', value: '下一交易日早晨9:00前更新' },
  { attribute: '数据处理', value: '中位数去极值 -> 行业市值对数中性化 -> zscore标准化' },
  { attribute: '默认参数', value: '加权方式为按市值加权' },
]);


const backtestParams = ref([
  { name: '组合构建', value: '', type: 'select', options: [{ label: '纯多头组合', value: 'long' }, { label: '多空组合', value: 'long_short' }] },
  { name: '股票池', value: '', type: 'select', options: [{ label: '中证500', value: 'zz500' }, { label: '沪深300', value: 'hs300' }, { label: '上证50', value: 'sz50' }, { label: '全A股', value: 'all_a' }] },
  { name: '回测区间', value: '', type: 'select', options: [{ label: '近1年', value: '1y' }, { label: '近2年', value: '2y' }, { label: '近3年', value: '3y' }, { label: '近5年', value: '5y' }] },
  { name: '过滤涨停及停牌股', value: '', type: 'select', options: [{ label: '是', value: 'yes' }, { label: '否', value: 'no' }] },
  { name: '调仓周期', value: '', type: 'select', options: [{ label: '1天', value: '1d' }, { label: '5天', value: '5d' }, { label: '10天', value: '10d' }, { label: '20天', value: '20d' }] },
  { name: '调仓时间', value: '', type: 'select', options: [{ label: '当天', value: 'same_day' }, { label: '15:00', value: '15:00' }] },
  { name: '手续费及滑点', value: '', type: 'select', options: [{ label: '无', value: 'none' }, { label: '0.1%', value: '0.1' }, { label: '0.3%', value: '0.3' }] },
  // "确定"按钮作为最后一个元素
  { type: 'button' }
]);


/*
const startBacktest = () => {
  console.log('开始回测');
};
*/

//定义后端IP地址
const backendUrl = 'http://192.168.43.135:8080/api/StockData';

const startBacktest = () => {
  axios.get(backendUrl)
    .then(response => {
      // 查询成功
      factorDetailsTableData.value = response.data;
      ElMessage.success('查询成功！');
    })
    .catch(error => {
      // 查询失败
      console.error(error);
      ElMessage.error('查询失败！');
    });
};

const factorDetailsTableData = ref([
 { data1:'累计收益',data2:'年化收益',data3:'超额年化收益',data4:'基准年化收益' ,data5:'最大回撤',data6:'夏普比率',data7:'换手率',},
{ data1:  '-27.15%',data2: '-10.33%',data3: '-1.82%',data4: '-8.68%',data5: '42.39%', data6: '-0.74',data7: '2.55%',},
  {data1: '-32.01%' ,data2: '-12.44%' , data3: '-4.12%', data4: '-8.68%', data5: '41.76%', data6: '-0.96', data7: '1.50%', }
]);

let yAxisType = ref('normal');
let showAllQuantiles = ref(false);

</script>

<script>
import * as echarts from 'echarts';

export default {
  data() {
    return {
      //因子详情图
      factorDetailsChartOptions: {
        title: {
          text: '{rect|} 因子详情',
          textStyle: {
            rich: {
              rect: {
                backgroundColor: 'rgb(180, 180, 180)',
                width: 10,
                height: 20,
                borderRadius: 2
              }
            }
          }
        },
        xAxis: {
          type: 'time',
          // 时间范围
          min: '2021-01-01',
          max: '2024-01-01',
          axisLabel: {
            formatter: function (value) {
              return echarts.format.formatTime('yyyy-MM-dd', value);
            }
          },
          splitLine: {
            show: true
          }
        },
        yAxis: {
          type: 'value',
          // 纵坐标范围
          min: -20,
          max: 20,
          axisLabel: {
            formatter: '{value}%',
            interval: 5
          },
          splitLine: {
            show: true
          }
        },
        series: [
        {
           // 待修改标题
          name: '中证 500',
          type: 'line',
          color: 'rgb(69, 114, 167)',
          data: [
            // 待填充数据0
            [new Date( '2021-01-01'), 5 ],
            [new Date( '2021-02-01'), 2 ],
            [new Date( '2021-03-01'), 10 ],
          ]
        },
        {
          name: '最小分位数',
          type: 'line',
          color: 'red',
          data: [
            // 待填充数据
            [new Date( '2021-01-01'), 10 ],
            [new Date( '2021-02-01'), -5 ],
            [new Date( '2021-03-01'), 15 ],
          ]
        },
        {
          name: '最大分位数',
          type: 'line',
          color: 'rgb(2, 144, 16)',
          data: [
            // 待填充数据
            [new Date( '2021-01-01'), -15 ],
            [new Date( '2021-02-01'), 6 ],
            [new Date( '2021-03-01'), -8 ],
          ]
        }
        ],
          // 通过提示框显示光标指向的数据
          tooltip: {
          trigger: 'axis',
          axisPointer: {
          type: 'cross'
          },
          formatter: function(params) {
            var tooltipContent = '';
            var xAxisValue = echarts.format.formatTime('yyyy-MM-dd', params[0].axisValue);

            tooltipContent += '时间：' + xAxisValue + '<br>';

            params.forEach(function(item) {
              tooltipContent += item.seriesName + ': ' + item.data[1] + '<br>';
            });

            return tooltipContent;
          }
        }
      },
      //IC时序图
      ICTimeSeriesChartOptions: {
        title: {
          text: '{rect|} IC时序图',
          textStyle: {
            rich: {
              rect: {
                backgroundColor: 'rgb(180, 180, 180)',
                width: 10,
                height: 20,
                borderRadius: 2
              }
            }
          }
        },
        xAxis: {
          type: 'time',
          // 时间范围
          min: '2021-01-01',
          max: '2024-01-01',
          axisLabel: {
            formatter: function (value) {
              return echarts.format.formatTime('yyyy-MM-dd', value);
            },
            interval: 2
          },
          splitLine: {
            show: true
          }
        },
        yAxis: {
          type: 'value',
          min: -20,
          max: 20,
          axisLabel: {
            formatter: '{value}%',
            interval: 5
          },
          splitLine: {
            show: true
          }
        },
        series: [
          {
            name: 'IC',
            type: 'line',
            color: 'rgb(124, 181, 236)',
            data: [
              // 待填充数据
              ['2021-01-01', 10],
              ['2021-02-01', 5],
              ['2021-03-01', -8],
            ]
          },
          {
            name: '22日移动平均',
            type: 'line',
            color: 'rgb(247, 163, 92)',
            data: [
              // 待填充数据
              ['2021-01-01', 2],
              ['2021-02-01', 1],
              ['2021-03-01', -1],
            ]
          }
        ],
          // 通过提示框显示光标指向的数据
          tooltip: {
          trigger: 'axis',
          axisPointer: {
          type: 'cross'
          },
          formatter: function(params) {
            var tooltipContent = '';
            var xAxisValue = echarts.format.formatTime('yyyy-MM-dd', params[0].axisValue);

            tooltipContent += '时间：' + xAxisValue + '<br>';

            params.forEach(function(item) {
              tooltipContent += item.seriesName + ': ' + item.data[1] + '<br>';
            });

            return tooltipContent;
          }
        }
        
      },
      // 行业IC图
      industryICChartOptions: {
        title: {
          text: '{rect|} 行业IC',
          textStyle: {
            rich: {
              rect: {
                backgroundColor: 'rgb(180, 180, 180)',
                width: 10,
                height: 20,
                borderRadius: 2
              }
            }
          }
        },
        xAxis: {
          type: 'value',
          min: -20,
          max: 20,
          axisLabel: {
            formatter: '{value}%',
            interval: 5
          },
          splitLine: {
            show: true
          }
        },
        yAxis: {
          type: 'category',
          data: ['能源', '材料', '工业', '可选消费', '日常消费', '医疗保健', '金融', '信息技术', '电信服务'] 
        },
        series: [
          {
            name: '行业IC',
            type: 'bar',
            stack: 'IC',
            //待填充数据
            data: [15, 8, 12, -10, -5, 4, -3, 20, -10], // 负数据
            itemStyle: {
              color: 'rgb(124, 181, 236)'
            }
          }
        ],
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'none' 
          },
          formatter: function(params) {
            var tooltipContent = '';
            var yAxisValue = params[0].axisValue;

            tooltipContent += '行业：' + yAxisValue + '<br>';

            params.forEach(function(item) {
              tooltipContent += item.seriesName + ': ' + item.data + '<br>';
            });

            return tooltipContent;
          }
        }
      }
    };
  },
  mounted() {
    this.initFactorDetailsChart();
    this.initICTimeSeriesChart();
    this.initIndustryICChart();
  },
  methods: {
    initFactorDetailsChart() {
      const chart = echarts.init(this.$refs.factorDetailsChart);
      chart.setOption(this.factorDetailsChartOptions);
    },
    initICTimeSeriesChart() {
      const chart = echarts.init(this.$refs.ICTimeSeriesChart);
      chart.setOption(this.ICTimeSeriesChartOptions);
    },
    initIndustryICChart() {
      const chart = echarts.init(this.$refs.industryICChart);
      chart.setOption(this.industryICChartOptions);
    }
  }
};
</script>


<style>
.factor-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 20px 0;
}

.param-item {
  display: flex;
  align-items: center;
  flex-wrap: nowrap;
}

.param-name {
  margin-right: 8px;
  white-space: nowrap;
}

.param-select {
  flex-grow: 1;
  min-width: 0;
}

.button-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
}

.el-menu {
  border-bottom: none;
}

.indicator-table {
  width: 80%;
  margin-top: 20px;
  margin-left: auto;
  margin-right: auto;
}

.indicator-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.indicator-name {
  font-size: 16px;
  font-weight: normal;
}

.indicator-value h1 {
  font-size: 2em;
  margin: 0;
}

.factor-details {
  margin-top: 20px;
}

.chart-title {
  text-align: center;
  margin-bottom: 20px;
}

.options-bar {
  display: flex;
  justify-content: center;
  margin-bottom: 20px;
}

.option-item {
  margin-right: 20px;
}

.line-chart {
  height: 400px; /* 根据需求调整 */
}
.factor-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 20px 0;
}
/* CSS */
.footer {
  background-color: #f0f0f0;
  padding: 20px 0;
  font-family: Arial, sans-serif;
  font-size: 14px;
  color: #333333;
}

.footer h4 {
  font-weight: bold;
  margin-bottom: 10px;
}

.footer p, .footer ul, .footer li {
  margin: 0;
  padding: 0;
  list-style: none;
}

.footer a {
  color: #333333;
  text-decoration: none;
}

.footer a:hover {
  color: #0099cc;
}

.footer .container {
  max-width: 960px;
  margin: 0 auto;
}

.footer .row {
  display: flex;
  flex-wrap: wrap;
}

.footer .col-md-4 {
  flex: 0 0 33.3333%;
  max-width: 33.3333%;
  padding: 0 15px;
}

.footer .copy {
  text-align: center;
  margin-top: 10px;
}

.chart-container {
  display: flex;
  flex-direction: column;
}

.chart-row {
  display: flex;
  flex-direction: row;
}

</style>
