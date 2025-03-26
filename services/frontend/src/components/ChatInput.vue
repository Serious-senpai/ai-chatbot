<script setup lang="ts">
import { ref } from "vue";

const props = defineProps(
  {
    onSubmit: {
      type: Function,
      required: true,
    },
  },
);

const input = ref("");

function submit(): void {
  const text = input.value.trim();
  input.value = "";
  props.onSubmit(text);
}
</script>

<template>
  <form @submit.prevent="submit">
    <input @submit.prevent="submit" v-model="input" class="bg-secondary d-block p-1 position-relative rounded-2 start-50 top-50 text-white translate-middle w-100" autocomplete="off" autocorrect="off" spellcheck="true" placeholder="Ask anything" type="text" />
  </form>
</template>

<style lang="css" scoped>
::placeholder {
  color: white;
  opacity: 1;
}
</style>
