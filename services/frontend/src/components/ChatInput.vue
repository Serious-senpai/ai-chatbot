<script setup lang="ts">
import { ref, useTemplateRef } from "vue";

const props = defineProps(
  {
    onSubmit: {
      type: Function,
      required: true,
    },
  },
);

const input = ref("");
const file = ref<File | null>(null);
const fileUpload = useTemplateRef("upload-file");

function uploadFile(event: Event): void {
  const target = event.target as HTMLInputElement;
  const files = target.files;
  if (files && files.length > 0) {
    file.value = files[0];
  } else {
    file.value = null;
  }
}

function uploadFileClick(): void {
  const fileInput = fileUpload.value as HTMLInputElement;
  if (fileInput) {
    fileInput.click();
  }
}

function submit(): void {
  const input_v = input.value.trim();
  const file_v = file.value;
  input.value = "";
  file.value = null;
  props.onSubmit(input_v, file_v);
}
</script>

<template>
  <form @submit.prevent="submit" class="d-flex flex-row">
    <input @click.prevent="uploadFileClick" class="d-inline-block p-1 rounded-1" style="width: 100px" :value="file?.name ?? 'Upload file'" type="button" />
    <input @change.prevent="uploadFile" accept=".pdf" class="d-none" ref="upload-file" type="file" />
    <input @submit.prevent="submit" v-model="input" class="bg-secondary d-inline-block flex-grow-1 p-1 rounded-2 text-white" autocomplete="off" autocorrect="off" spellcheck="true" placeholder="Ask anything" type="text" />
  </form>
</template>

<style lang="css" scoped>
::placeholder {
  color: white;
  opacity: 1;
}
</style>
