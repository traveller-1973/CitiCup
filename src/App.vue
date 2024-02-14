<template>
  <div>
    <el-menu default-active="1" class="el-menu-demo" mode="horizontal" style="width: 100%">
      <el-menu-item index="1">Home</el-menu-item>
      <el-menu-item index="2">About</el-menu-item>
      <el-menu-item index="3">Contact</el-menu-item>
    </el-menu>
     <el-row>
    <el-col :span="2"><div class="grid-content ep-bg-purple-light" /></el-col>

    <el-col :span="20"><div class="grid-content ep-bg-purple" />
<h3>因子看板</h3>
      <br>
      <!--    分类栏-->
    <el-row>
    <el-col :span="2"><div class="grid-content ep-bg-purple" >分类：</div></el-col>
    <el-col :span="2"><div class="grid-content ep-bg-purple" ><el-checkbox
      v-model="checkAll"
      :indeterminate="isIndeterminate"
      @change="handleCheckAllChange"
      >全选</el-checkbox></div></el-col>

    <el-col :span="20"><div class="grid-content ep-bg-purple-light" >
    <el-checkbox-group
      v-model="checkedCities"
      @change="handleCheckedCitiesChange"
    >
      <el-checkbox v-for="city in cities" :key="city" :label="city">{{ city }}</el-checkbox>
    </el-checkbox-group></div></el-col>
  </el-row>
<el-row>
<!--  股票池-->
    <el-col :span="2"><div class="grid-content ep-bg-purple" >股票池：</div></el-col>


    <el-col :span="22"><div class="grid-content ep-bg-purple-light" >
    <!--    单选-->
     <div class="mb-2 flex items-center text-sm">
    <el-radio-group v-model="radio1" class="ml-4">
      <el-radio label="1" size="large">沪深300</el-radio>
      <el-radio label="2" size="large">中证500</el-radio>
      <el-radio label="3" size="large">中证800</el-radio>
      <el-radio label="4" size="large">中证1000</el-radio>
      <el-radio label="5" size="large">中证全指</el-radio>

    </el-radio-group>
  </div>
    </div></el-col>
  </el-row>
<!--  回测周期-->
<el-row>
    <el-col :span="2"><div class="grid-content ep-bg-purple" >回测周期：</div></el-col>


    <el-col :span="22"><div class="grid-content ep-bg-purple-light" >
    <!--    单选-->
     <div class="mb-2 flex items-center text-sm">
    <el-radio-group v-model="radio2" class="ml-4">
      <el-radio label="1" size="large">近3个月</el-radio>
      <el-radio label="2" size="large">近1年</el-radio>
      <el-radio label="3" size="large">近3年</el-radio>
      <el-radio label="4" size="large">近10年</el-radio>

    </el-radio-group>
  </div>
    </div></el-col>
  </el-row>
    <!--  组合构建-->
<el-row>
    <el-col :span="2"><div class="grid-content ep-bg-purple" >组合构建：</div></el-col>


    <el-col :span="22"><div class="grid-content ep-bg-purple-light" >
    <!--    单选-->
     <div class="mb-2 flex items-center text-sm">
    <el-radio-group v-model="radio3" class="ml-4">
      <el-radio label="1" size="large">纯多头组合</el-radio>
      <el-radio label="2" size="large">多空组合1</el-radio>
      <el-radio label="3" size="large">多空组合2</el-radio>

    </el-radio-group>
  </div>
    </div></el-col>
  </el-row>

     <!--  手续费-->
<el-row>
    <el-col :span="2"><div class="grid-content ep-bg-purple" >手续费及滑点：</div></el-col>


    <el-col :span="22"><div class="grid-content ep-bg-purple-light" >
    <!--    单选-->
     <div class="mb-2 flex items-center text-sm">
    <el-radio-group v-model="radio4" class="ml-4">
      <el-radio label="1" size="large">无</el-radio>
      <el-radio label="2" size="large">3%%佣金+1%%印花税+无滑点</el-radio>
      <el-radio label="3" size="large">3%%佣金+1%%印花税+1%%无滑点</el-radio>

    </el-radio-group>
  </div>
    </div></el-col>
  </el-row>

     <!--  过滤涨停-->
<el-row>
    <el-col :span="2"><div class="grid-content ep-bg-purple" >过滤涨停及停牌股：</div></el-col>


    <el-col :span="22"><div class="grid-content ep-bg-purple-light" >
    <!--    单选-->
     <div class="mb-2 flex items-center text-sm">
    <el-radio-group v-model="radio5" class="ml-4">
      <el-radio label="1" size="large">否</el-radio>
      <el-radio label="2" size="large">是</el-radio>

    </el-radio-group>
  </div>
    </div></el-col>
  </el-row>
      <br>
    <div class="container">
      <el-table :data="tableData" border style="width: 100%">
        <el-table-column prop="因子名称" label="因子名称" width="180" />
        <el-table-column prop="最小分数超额年化收益率" label="最小分数超额年化收益率" width="180" />
        <el-table-column prop="最小分数换手率" label="最小分数换手率" />
        <el-table-column prop="最大分数换手率" label="最大分数换手率" />
        <el-table-column prop="IC均值" label="IC均值" />
        <el-table-column prop="IR均值" label="IR均值" />

      </el-table>
    </div></el-col>
    <el-col :span="2"><div class="grid-content ep-bg-purple-light" /></el-col>
  </el-row>
    <br>

  </div>
</template>

<script setup>
import { ref } from 'vue';

// 多项选择
const checkAll = ref(false)
const isIndeterminate = ref(true)
const checkedCities = ref(['Shanghai', 'Beijing'])
const cities = ['基础科目及其衍生类因子', '质量类因子', '每股指标因子', '风险因子-风格因子','情绪类因子','成长类因子','风险类因子','技术指标因子','动量类因子','风险因子-新风格因子']

const handleCheckAllChange = (val) => {
  checkedCities.value = val ? cities : []
  isIndeterminate.value = false
}
const handleCheckedCitiesChange = (value) => {
  const checkedCount = value.length
  checkAll.value = checkedCount === cities.length
  isIndeterminate.value = checkedCount > 0 && checkedCount < cities.length
}
// 单项选择-股票池
const radio1 = ref('1')
// 单项选择-回测周期
const radio2 = ref('2')
// 单项选择-组合构建
const radio3 = ref('3')
// 单项选择-手续费
const radio4 = ref('4')
// 单项选择-过滤涨停
const radio5 = ref('5')

// 表单数据
const tableData = [

  {
    因子名称: '2016-05-02',
    最小分数超额年化收益率: 'Tom',
    最小分数换手率: 'No. 189, Grove St, Los Angeles',
    最大分数换手率: 'No. 189, Grove St, Los Angeles',
    IC均值: 'No. 189, Grove St, Los Angeles',
    IR均值: 'No. 189, Grove St, Los Angeles',

  },

]
</script>

<style>
.container {
  margin: 20px auto;
}
.el-menu-demo {
  background-color: #000080; /* 深蓝色 */
  color: #fff; /* 文字颜色 */
}
</style>
